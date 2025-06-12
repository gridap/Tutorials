# # [Mixed-order Hybrid High-Order Method for the Poisson equation](@id poisson_hho)
#
# In this tutorial, we will learn how to implement a mixed-order Hybrid High-Order (HHO) method
# for solving the Poisson equation. HHO methods are a class of modern hybridizable finite element methods
# that provide optimal convergence rates while enabling static condensation for efficient solution.
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
# where Ω is a bounded domain in R², f is a source term, and g is the prescribed boundary value.
#
# ## HHO discretization
#
# The HHO method introduces two types of unknowns:
# 1. Cell unknowns defined in the volume of each mesh cell
# 2. Face unknowns defined on the mesh faces/edges
#
# This hybrid structure allows for efficient static condensation by eliminating the cell unknowns
# algebraically at the element level.
#
# Load the required packages

using Gridap
using Gridap.Geometry, Gridap.FESpaces, Gridap.MultiField
using Gridap.CellData, Gridap.Fields, Gridap.Helpers
using Gridap.ReferenceFEs
using Gridap.Arrays

# ## Local projection operator
# 
# Define a projection operator to map functions onto local polynomial spaces

function projection_operator(V, Ω, dΩ)
  Π(u,Ω) = change_domain(u,Ω,DomainStyle(u))
  mass(u,v) = ∫(u⋅Π(v,Ω))dΩ
  V0 = FESpaces.FESpaceWithoutBCs(V)
  P = LocalOperator(
    LocalSolveMap(), V0, mass, mass; trian_out = Ω
  )
  return P
end

# ## Reconstruction operator
#
# Define a reconstruction operator that maps hybrid unknowns to a higher-order polynomial space.
# This operator is key for achieving optimal convergence rates.

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

# ## Problem setup
#
# Define the exact solution and forcing term

u(x) = sin(2*π*x[1])*sin(2*π*x[2])
f(x) = -Δ(u)(x)

# Setup the mesh and discretization parameters

n = 10
base_model = simplexify(CartesianDiscreteModel((0,1,0,1),(n,n)))
model = Geometry.voronoi(base_model)

D = num_cell_dims(model)
Ω = Triangulation(ReferenceFE{D}, model)
Γ = Triangulation(ReferenceFE{D-1}, model)

ptopo = Geometry.PatchTopology(model)
Ωp = Geometry.PatchTriangulation(model,ptopo)
Γp = Geometry.PatchBoundaryTriangulation(model,ptopo)

order = 1
qdegree = 2*(order+1)

dΩp = Measure(Ωp,qdegree)
dΓp = Measure(Γp,qdegree)

# ## FE spaces and operators
#
# Define the finite element spaces for cell and face unknowns

V = FESpaces.PolytopalFESpace(Ω, Float64, order+1; space=:P) # Bulk space
M = FESpaces.PolytopalFESpace(Γ, Float64, order; space=:P, dirichlet_tags="boundary") # Skeleton space
N = TrialFESpace(M,u)

mfs = MultiField.BlockMultiFieldStyle(2,(1,1))
X   = MultiFieldFESpace([V, N];style=mfs)
Y   = MultiFieldFESpace([V, M];style=mfs) 
Xp  = FESpaces.PatchFESpace(X,ptopo)

# Setup projection and reconstruction operators

PΓ = projection_operator(M, Γp, dΓp)
R  = reconstruction_operator(ptopo,order,Y,Ωp,Γp,dΩp,dΓp)

# Setup assemblers

global_assem = SparseMatrixAssembler(X,Y)
patch_assem = FESpaces.PatchAssembler(ptopo,X,Y)

# ## Bilinear and linear forms
#
# Define the bilinear form a(u,v) for the diffusion term

function a(u,v)
  Ru_Ω, Ru_Γ = R(u)
  Rv_Ω, Rv_Γ = R(v)
  return ∫(∇(Ru_Ω)⋅∇(Rv_Ω) + ∇(Ru_Γ)⋅∇(Rv_Ω) + ∇(Ru_Ω)⋅∇(Rv_Γ) + ∇(Ru_Γ)⋅∇(Rv_Γ))dΩp
end

# Compute the inverse of local cell measure for stabilization

hTinv = CellField(1 ./ collect(get_array(∫(1)dΩp)), Ωp)

# Define the stabilization term s(u,v) to weakly enforce continuity

function s(u,v)
  function SΓ(u)
    u_Ω, u_Γ = u
    return PΓ(u_Ω) - u_Γ
  end
  return ∫(hTinv * (SΓ(u)⋅SΓ(v)))dΓp
end

# Define the linear form l(v) for the source term

l((vΩ,vΓ)) = ∫(f⋅vΩ)dΩp

# ## Problem solution
#
# Set up the weak form and solve using direct or static condensation

function weakform()
  u, v = get_trial_fe_basis(X), get_fe_basis(Y)
  data = FESpaces.collect_and_merge_cell_matrix_and_vector(
    (Xp, Xp, a(u,v), DomainContribution(), zero(Xp)),
    (X, Y, s(u,v), l(v), zero(X))
  )
  assemble_matrix_and_vector(global_assem,data)
end

function patch_weakform()
  u, v = get_trial_fe_basis(X), get_fe_basis(Y)
  data = FESpaces.collect_and_merge_cell_matrix_and_vector(patch_assem,
    (Xp, Xp, a(u,v), DomainContribution(), zero(Xp)),
    (X, Y, s(u,v), l(v), zero(X))
  )
  return assemble_matrix_and_vector(patch_assem,data)
end

# Direct monolithic solve

A, b = weakform()
x = A \ b

ui, ub = FEFunction(X,x)
eu  = ui - u 
l2u = sqrt(sum( ∫(eu * eu)dΩp))
h1u = l2u + sqrt(sum( ∫(∇(eu) ⋅ ∇(eu))dΩp))

# Static condensation

op = MultiField.StaticCondensationOperator(X,patch_assem,patch_weakform())
ui, ub = solve(op) 

eu  = ui - u
l2u = sqrt(sum( ∫(eu * eu)dΩp))
h1u = l2u + sqrt(sum( ∫(∇(eu) ⋅ ∇(eu))dΩp))

# The code above demonstrates both solution approaches:
#
# 1. Direct solution of the full system
# 2. Static condensation to eliminate cell unknowns
#
# Both give the same solution but static condensation is typically more efficient
# for higher orders.
