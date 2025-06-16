#
# In this tutorial, we will learn how to implement a mixed-order Hybrid High-Order (HHO) method
# for solving the Poisson equation. HHO methods are a class of modern hybridizable finite element methods
# that provide optimal convergence rates while enabling static condensation for efficient solution.
#
# HHO is a mathematically complex method. This tutorial will **NOT** cover the method nor 
# its mathematical foundations. You should be familiar with both before going into 
# the tutorial itself. 
# We also recommend going through the HDG tutorial first, which shares many of the 
# same concepts but is simpler to understand.
#
# ## Problem statement
#
# We consider the Poisson equation with Dirichlet boundary conditions:
#
# ```math
# \begin{aligned}
# -\Delta u &= f \quad \text{in } \Omega \\
# u &= g \quad \text{on } \partial\Omega
# \end{aligned}
# ```
#
# where $\Omega$ is a bounded domain in $\mathbb{R}^2$, f is a source term, and $g$ is the prescribed boundary value.
#
# ## HHO discretization
#
# The HHO method introduces two types of unknowns:
# 1. Cell unknowns defined in the volume of each mesh cell
# 2. Face unknowns defined on the mesh facets
#
# This hybrid structure allows for efficient static condensation by eliminating the cell unknowns
# algebraically at the element level.
#
# We start by loading the required packages

using Gridap
using Gridap.Geometry, Gridap.FESpaces, Gridap.MultiField
using Gridap.CellData, Gridap.Fields, Gridap.Helpers
using Gridap.ReferenceFEs
using Gridap.Arrays

# ## Geometry
#
# We generate a 2-dimensional simplicial mesh from a Cartesian grid:

u(x) = sin(2*π*x[1])*sin(2*π*x[2])
f(x) = -Δ(u)(x)

n = 10
base_model = simplexify(CartesianDiscreteModel((0,1,0,1),(n,n)))
model = Geometry.voronoi(base_model)

# From this mesh, we will require two triangulations where to define our HDG spaces:
#
# 1. A cell triangulation $\Omega$, for the volume variables
# 2. A face triangulation $\Gamma$, for the skeleton variables
#
# These are given by

D = num_cell_dims(model)
Ω = Triangulation(ReferenceFE{D}, model)
Γ = Triangulation(ReferenceFE{D-1}, model)

# ## FESpaces
#
# HHO uses two different finite element spaces:
# 1. A scalar space for the bulk variable uT (V)
# 2. A scalar space for the skeleton variable uF (M)
#
# We then define discontinuous finite element spaces of the approppriate order, locally $$\mathbb{P}^k$$.
# Because we are using a mixed-order scheme, the bulk space has a higher polynomial order
# than the skeleton space. 
# Note that only the skeletal space has Dirichlet boundary conditions.

order = 1
V = FESpaces.PolytopalFESpace(Ω, Float64, order+1; space=:P) # Bulk space
M = FESpaces.PolytopalFESpace(Γ, Float64, order; space=:P, dirichlet_tags="boundary") # Skeleton space
N = TrialFESpace(M,u)

# ## MultiField Structure
#
# Since we are doing static condensation, we need assemble by blocks. In particular, the 
# `StaticCondensationOperator` expects the variables to be groupped in two blocks:
#   - The eliminated variables (in this case, the volume variables q and u)
#   - The retained variables (in this case, the interface variable m)
# We will assemble by blocks using the `BlockMultiFieldStyle` API. 

mfs = MultiField.BlockMultiFieldStyle(2,(1,1))
X   = MultiFieldFESpace([V, N];style=mfs)
Y   = MultiFieldFESpace([V, M];style=mfs) 

# ## PatchTopology and PatchTriangulation
#
# A key aspect of hybrid methods is the use of static condensation, which is the 
# elimination of cell unknowns to reduce the size of the global system.
# To achieve this, we need to be able to assemble and solve local problems on each cell, that 
# involve 
#   - contributions from the cell itself
#   - contributions from the cell faces
# To this end, Gridap provides a general framework for patch-assembly and solves. The idea 
# is to define a patch decomposition of the mesh (in this case, a patch is a cell and its sourrounding 
# faces). We can then gather contributions for each patch, solve the local problems, and 
# assemble the results into the global system.
#
# The following code creates the required `PatchTopology` for the problem at hand. We then 
# take d-dimensional slices of it by the means of `PatchTriangulation` and `PatchBoundaryTriangulation`.
# These are the `Triangulation`s we will integrate our weakform over.

ptopo = Geometry.PatchTopology(model)
Ωp = Geometry.PatchTriangulation(model,ptopo)
Γp = Geometry.PatchBoundaryTriangulation(model,ptopo)

qdegree = 2*(order+1)
dΩp = Measure(Ωp,qdegree)
dΓp = Measure(Γp,qdegree)

# ## Local operators
# 
# A key feature of HHO is the use of local solves to define local projections of our bulk and 
# skeleton variables. Just like for static condensation, we will use patch assembly to 
# gather the contributions from each patch and solve the local problems.
# The result is then an element of the space we are projecting to, given as a 
# linear combination of the basis of that space. This whole abstraction is taken 
# care of by the `LocalOperator` object. 
#
# For the mixed-order Poisson problem, we require two local projections. 
#
# ### L2 projection operator
#
# First, an L2 local projection operator onto the mesh faces. This is the simplest 
# operator we can define, such that given $u$ in some undefined space, we find $Pu \in V$ st
#
# $$(Pu,q) = (u,q) \forall q \in V$$
#
# This signature for the `LocalOperator` assumes that the rhs and lhs for each local problem
# are given by a single cell contribution (no patch assembly required).
# Note also the use of `FESpaceWithoutBCs`, which strips the boundary conditions from the 
# space `V`. This is because we do not want to take into account boundary conditions 
# when projecting onto the space.
# The local solve map is given by `LocalSolveMap`, which by default uses Julia's LU 
# factorization to solve the local problem exactly.

function projection_operator(V, Ω, dΩ)
  Π(u,Ω) = change_domain(u,Ω,DomainStyle(u))
  mass(u,v) = ∫(u⋅Π(v,Ω))dΩ
  V0 = FESpaces.FESpaceWithoutBCs(V)
  P = LocalOperator(
    LocalSolveMap(), V0, mass, mass; trian_out = Ω
  )
  return P
end

# ### Reconstruction operator
# 
# Finally, we build the so-called reconstruction operator. This operator is highly tied to the 
# ellipic projector, and projects our bulk-skeleton variable pair onto a bulk space of higher order.
#
# It's definition can be found in HHO literature, and requires solving a constrained
# local problem on each cell and its faces. We therefore use patch assembly, and impose 
# our constraint using an additional space `Λ` as a local Lagrange multiplier. 
# Naturally, we want to eliminate the multiplier and return a solution in the reconstructed 
# space `L`. This is taken care of by the `LocalPenaltySolveMap`, and the kwarg `space_out = L`
# which overrides the default behavior of returning a solution in the test space `Y`.

function reconstruction_operator(ptopo,order,X,Ω,Γp,dΩp,dΓp)
  L = FESpaces.PolytopalFESpace(Ω, Float64, order+1; space=:P)
  Λ = FESpaces.PolytopalFESpace(Ω, Float64, 0; space=:P)

  n = get_normal_vector(Γp)
  Πn(v) = ∇(v)⋅n
  Π(u,Ω) = change_domain(u,Ω,DomainStyle(u))
  lhs((u,λ),(v,μ))   = ∫( (∇(u)⋅∇(v)) + (μ*u) + (λ*v) )dΩp
  rhs((uT,uF),(v,μ)) =  ∫( (∇(uT)⋅∇(v)) + (uT*μ) )dΩp + ∫( (uF - Π(uT,Γp))*(Πn(v)) )dΓp
  
  Y = FESpaces.FESpaceWithoutBCs(X)
  W = MultiFieldFESpace([L,Λ];style=BlockMultiFieldStyle())
  R = LocalOperator(
    LocalPenaltySolveMap(), ptopo, W, Y, lhs, rhs; space_out = L
  )
  return R
end

PΓ = projection_operator(M, Γp, dΓp)
R  = reconstruction_operator(ptopo,order,Y,Ωp,Γp,dΩp,dΓp)

# ## Weakform and assembly
#
# We can now define: 
#   - The consistency term `a`
#   - The stabilization term `s`
#   - The rhs term `l`

hTinv = CellField(1 ./ collect(get_array(∫(1)dΩp)), Ωp)

function a(u,v)
  Ru_Ω, Ru_Γ = R(u)
  Rv_Ω, Rv_Γ = R(v)
  return ∫(∇(Ru_Ω)⋅∇(Rv_Ω) + ∇(Ru_Γ)⋅∇(Rv_Ω) + ∇(Ru_Ω)⋅∇(Rv_Γ) + ∇(Ru_Γ)⋅∇(Rv_Γ))dΩp
end

function s(u,v)
  function SΓ(u)
    u_Ω, u_Γ = u
    return PΓ(u_Ω) - u_Γ
  end
  return ∫(hTinv * (SΓ(u)⋅SΓ(v)))dΓp
end

l((vΩ,vΓ)) = ∫(f⋅vΩ)dΩp

# ### Patch-FESpaces
#
# An additional difficulty in HHO methods is that our reconstructed functions `R(u)` are 
# hard to assemble. They are defined on the cells, but depend on skeleton degrees of freedom. 
# We therefore cannot assemble them using the original FESpace `N`. Instead, we will create \
# view of the original space that is defined on the patches, and can be used to assemble 
# the local contributions depending on `R`. It can be done by the means of `PatchFESpace`:

Xp  = FESpaces.PatchFESpace(X,ptopo)

# ### Assembly without static condensation
#
# We can now proceed to evaluate and assemble all our contributions. Note that some of 
# the contributions depend on the reconstructed variables, e.g `a(u,v)`, and need to 
# be assembled using the `PatchFESpace` `Xp`, while others can be assembled using the original 
# FESpace `X` (e.g. `s(u,v)` and `l(v)`). 
# Assembly is therefore somewhat more complex than in the standard case. We need to 
# collect the contributions for every (test,trial) pair, and then merge them into a single
# data structure that can be passed to the assembler.
# In the case of mixed-order HHO, we only have two combinations, (`Xp`, `Xp`) and (`X`, `X`), 
# but the cross-terms appear also in the case of the original HHO formulation. 

global_assem = SparseMatrixAssembler(X,Y)

function weakform()
  u, v = get_trial_fe_basis(X), get_fe_basis(Y)
  data = FESpaces.collect_and_merge_cell_matrix_and_vector(
    (Xp, Xp, a(u,v), DomainContribution(), zero(Xp)),
    (X, Y, s(u,v), l(v), zero(X))
  )
  assemble_matrix_and_vector(global_assem,data)
end

A, b = weakform()
x = A \ b

ui, ub = FEFunction(X,x)
eu  = ui - u 
l2u = sqrt(sum( ∫(eu * eu)dΩp))
h1u = l2u + sqrt(sum( ∫(∇(eu) ⋅ ∇(eu))dΩp))

# ### Assembly with static condensation

patch_assem = FESpaces.PatchAssembler(ptopo,X,Y)

function patch_weakform()
  u, v = get_trial_fe_basis(X), get_fe_basis(Y)
  data = FESpaces.collect_and_merge_cell_matrix_and_vector(patch_assem,
    (Xp, Xp, a(u,v), DomainContribution(), zero(Xp)),
    (X, Y, s(u,v), l(v), zero(X))
  )
  return assemble_matrix_and_vector(patch_assem,data)
end

op = MultiField.StaticCondensationOperator(X,patch_assem,patch_weakform())
ui, ub = solve(op) 

eu  = ui - u
l2u = sqrt(sum( ∫(eu * eu)dΩp))
h1u = l2u + sqrt(sum( ∫(∇(eu) ⋅ ∇(eu))dΩp))

# ## Going further
#
# This tutorial has introduced the basic concepts of HHO methods using the simplest their
# simplest form, e.g. mixed-order HHO for the Poisson equation.
# More advanced drivers can be found with Gridap's tests. In particular:
#
#   - [Poisson with original HHO formulation](https://github.com/gridap/Gridap.jl/blob/75efc9a7a7e286c27e7ca3ddef5468e591845484/test/GridapTests/HHOPolytopalTests.jl)
#   - [Incompressible Stokes](https://github.com/gridap/Gridap.jl/blob/75efc9a7a7e286c27e7ca3ddef5468e591845484/test/GridapTests/HHOMixedStokesPolytopal.jl) 
#   - [Linear Elasticity](https://github.com/gridap/Gridap.jl/blob/75efc9a7a7e286c27e7ca3ddef5468e591845484/test/GridapTests/HHOMixedElasticity.jl)
#
