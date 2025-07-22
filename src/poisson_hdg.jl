# In this tutorial, we will implement a Hybridizable Discontinuous Galerkin (HDG) method
# for solving the Poisson equation. The HDG method is an efficient variant of DG methods
# that introduces an auxiliary variable m on mesh interfaces to reduce the global system size.
#
# ## HDG Discretization
#
# We consider the Poisson equation with Dirichlet boundary conditions:
#
# ```math
# \begin{aligned}
# -\Delta u &= f \quad \text{in} \quad \Omega\\
# u &= g \quad \text{in} \quad \partial\Omega
# \end{aligned}
# ```
#
# The HDG method first rewrites the problem as a first-order system:
#
# ```math
# \begin{aligned}
# \boldsymbol{q} + \nabla u &= \boldsymbol{0} \quad \text{in} \quad \Omega\\
# \nabla \cdot \boldsymbol{q} &= f \quad \text{in} \quad \Omega\\
# u &= g \quad \text{on} \quad \partial\Omega
# \end{aligned}
# ```
#
# The HDG discretization introduces three variables:
# - ``\boldsymbol{q}_h``: the approximation to the flux ``\boldsymbol{q}``
# - ``u_h``: the approximation to the solution ``u``
# - ``m_h``: the approximation to the trace of ``u`` on element faces
#
# Numerical fluxes are defindes as 
#
# ```math
# \widehat{\boldsymbol{q}}_h = \boldsymbol{q}_h + \tau(u_h - m_h)\boldsymbol{n}
# ```
#
# where $\tau$ is a stabilization parameter.
#
# First, let's load the required Gridap packages:

using Gridap
using Gridap.Geometry
using Gridap.FESpaces
using Gridap.MultiField
using Gridap.CellData

# ## Manufactured Solution
#
# We use the method of manufactured solutions to verify our implementation.
# We choose a solution u and derive the corresponding source term f:

u(x) = sin(2*π*x[1])*sin(2*π*x[2])
q(x) = -∇(u)(x)  # Define the flux q = -∇u
f(x) = (∇ ⋅ q)(x) # Source term f = -Δu = -∇⋅(∇u)$

# ## Geometry
#
# We generate a D-dimensional simplicial mesh from a Cartesian grid:

D = 2  # Problem dimension
nc = Tuple(fill(8, D))  # 4 cells in each direction
domain = Tuple(repeat([0, 1], D))  # Unit cube domain
model = simplexify(CartesianDiscreteModel(domain,nc))

# From this mesh, we will require two triangulations where to define our HDG spaces:
#
# 1. A cell triangulation $\Omega$, for the volume variables
# 2. A face triangulation $\Gamma$, for the skeleton variables
#
# These are given by

Ω = Triangulation(ReferenceFE{D}, model)  # Volume triangulation
Γ = Triangulation(ReferenceFE{D-1}, model)  # Skeleton triangulation

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
Ωp = Geometry.PatchTriangulation(model,ptopo)  # Patch volume triangulation
Γp = Geometry.PatchBoundaryTriangulation(model,ptopo)  # Patch skeleton triangulation

# ## FESpaces
#
# HDG uses three different finite element spaces:
# 1. A vector-valued space for the flux q (Q)
# 2. A scalar space for the solution u (V)
# 3. A scalar space for the interface variable m (M)
#
# We then define discontinuous finite element spaces of the approppriate order, locally $\mathbb{P}^k$.
# Note that only the skeletal space has Dirichlet boundary conditions.

order = 1  # Polynomial order
reffe_Q = ReferenceFE(lagrangian, VectorValue{D, Float64}, order; space=:P)
reffe_V = ReferenceFE(lagrangian, Float64, order; space=:P)
reffe_M = ReferenceFE(lagrangian, Float64, order; space=:P)

V = TestFESpace(Ω, reffe_V; conformity=:L2)  # Discontinuous vector space
Q = TestFESpace(Ω, reffe_Q; conformity=:L2)  # Discontinuous scalar space
M = TestFESpace(Γ, reffe_M; conformity=:L2, dirichlet_tags="boundary")  # Interface space
N = TrialFESpace(M, u)

# ## MultiField Structure
#
# Since we are doing static condensation, we need assemble by blocks. In particular, the 
# `StaticCondensationOperator` expects the variables to be groupped in two blocks:
#   - The eliminated variables (in this case, the volume variables q and u)
#   - The retained variables (in this case, the interface variable m)
# We will assemble by blocks using the `BlockMultiFieldStyle` API. 

mfs = BlockMultiFieldStyle(2,(2,1))  # Special blocking for efficient static condensation
X = MultiFieldFESpace([V, Q, N]; style=mfs)

# ## Weak Form and integration
#

degree = 2*(order+1)  # Integration degree
dΩp = Measure(Ωp,degree)  # Volume measure, on the patch triangulation
dΓp = Measure(Γp,degree)  # Surface measure, on the patch boundary triangulation

τ = 1.0 # HDG stabilization parameter

n = get_normal_vector(Γp)  # Face normal vector
Πn(u) = u⋅n  # Normal component
Π(u) = change_domain(u,Γp,DomainStyle(u))  # Project to skeleton

a((uh,qh,sh),(vh,wh,lh)) = ∫( qh⋅wh - uh*(∇⋅wh) - qh⋅∇(vh) )dΩp + ∫(sh*Πn(wh))dΓp +
                           ∫((Πn(qh) + τ*(Π(uh) - sh))*(Π(vh) + lh))dΓp
l((vh,wh,lh)) = ∫( f*vh )dΩp

# ## Static Condensation and Solution
#
# With all these ingredients, we can now build our statically-condensed operator ans 
# solve the problem. Note that we are solving a scatically-condensed system. We can 
# retrieve the internal `AffineFEOperator` from `op.sc_op`.

op = MultiField.StaticCondensationOperator(ptopo,X,a,l)
uh, qh, sh = solve(op)

dΩ = Measure(Ω,degree)
eh = uh - u
l2_uh = sqrt(sum(∫(eh⋅eh)*dΩ))

mkpath("output_path")
writevtk(Ω,"output_path/results",cellfields=["uh"=>uh,"qh"=>qh,"eh"=>eh])

# ## Going Further
#
# By modifying the stabilisation term, HDG can also work on polytopal meshes. An driver 
# solving the same problem on a polytopal mesh is available [in the Gridap repository](https://github.com/gridap/Gridap.jl/blob/75efc9a7a7e286c27e7ca3ddef5468e591845484/test/GridapTests/HDGPolytopalTests.jl). 
# A tutorial for HHO on polytopal meshes is also available. 
#
