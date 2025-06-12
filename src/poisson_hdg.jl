# # [Tutorial: Hybridizable Discontinuous Galerkin Method for the Poisson equation]
#
# In this tutorial, we will implement a Hybridizable Discontinuous Galerkin (HDG) method
# for solving the Poisson equation. The HDG method is an efficient variant of DG methods
# that introduces an auxiliary variable λ on mesh interfaces to reduce the global system size.
#
# ## The Poisson Problem
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
# ## HDG Discretization
#
# The HDG method first rewrites the problem as a first-order system:
#
# ```math
# \begin{aligned}
# \boldsymbol{q} + \nabla u &= \boldsymbol{0} \quad \text{in} \quad \Omega\\
# \nabla \cdot \boldsymbol{q} &= -f \quad \text{in} \quad \Omega\\
# u &= g \quad \text{on} \quad \partial\Omega
# \end{aligned}
# ```
#
# The HDG discretization introduces three variables:
# - ``\boldsymbol{q}_h``: the approximation to the flux ``\boldsymbol{q}``
# - ``u_h``: the approximation to the solution ``u``
# - ``\lambda_h``: the approximation to the trace of ``u`` on element faces
#
# The method is characterized by:
# 1. Local discontinuous approximation for ``(\boldsymbol{q}_h,u_h)`` in each element
# 2. Continuous approximation for ``\lambda_h`` on element faces
# 3. Numerical flux ``\widehat{\boldsymbol{q}}_h = \boldsymbol{q}_h + \tau(u_h - \lambda_h)\boldsymbol{n}``
#
# where τ is a stabilization parameter.
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
f(x) = (∇ ⋅ q)(x) # Source term f = -Δu = -∇⋅(∇u)

# ## Mesh Generation
#
# Create a 3D simplicial mesh from a Cartesian grid:

D = 2  # Problem dimension
nc = Tuple(fill(8, D))  # 4 cells in each direction
domain = Tuple(repeat([0, 1], D))  # Unit cube domain
model = simplexify(CartesianDiscreteModel(domain,nc))

# ## Volume and Interface Triangulations
#
# HDG methods require two types of meshes:
# 1. Volume mesh for element interiors
# 2. Skeleton mesh for element interfaces
#
# We also need patch-wise triangulations for local computations:

Ω = Triangulation(ReferenceFE{D}, model)  # Volume triangulation
Γ = Triangulation(ReferenceFE{D-1}, model)  # Skeleton triangulation

ptopo = Geometry.PatchTopology(model)
Ωp = Geometry.PatchTriangulation(model,ptopo)  # Patch volume triangulation
Γp = Geometry.PatchBoundaryTriangulation(model,ptopo)  # Patch skeleton triangulation

# ## Reference Finite Elements
#
# HDG uses three different finite element spaces:
# 1. Vector-valued space for the flux q (Q)
# 2. Scalar space for the solution u (V)
# 3. Scalar space for the interface variable λ (M)

order = 1  # Polynomial order
reffe_Q = ReferenceFE(lagrangian, VectorValue{D, Float64}, order; space=:P)
reffe_V = ReferenceFE(lagrangian, Float64, order; space=:P)
reffe_M = ReferenceFE(lagrangian, Float64, order; space=:P)

# ## Test and Trial Spaces
#
# Create discontinuous cell spaces for volume variables (q,u) and 
# a discontinous face space for the interface variable λ:

# Test spaces
V = TestFESpace(Ω, reffe_V; conformity=:L2)  # Discontinuous vector space
Q = TestFESpace(Ω, reffe_Q; conformity=:L2)  # Discontinuous scalar space
M = TestFESpace(Γ, reffe_M; conformity=:L2, dirichlet_tags="boundary")  # Interface space

# Only the skeleton space has Dirichlet BC
N = TrialFESpace(M, u)

# ## MultiField Structure
#
# Group the spaces for q, u, and λ using MultiField:

mfs = MultiField.BlockMultiFieldStyle(2,(2,1))  # Special blocking for efficient static condensation
X = MultiFieldFESpace([V, Q, N];style=mfs)

# ## Integration Setup
#
# Define measures for volume and face integrals:

degree = 2*(order+1)  # Integration degree
dΩp = Measure(Ωp,degree)  # Volume measure
dΓp = Measure(Γp,degree)  # Surface measure

# ## HDG Parameters
#
# The stabilization parameter τ affects the stability and accuracy of the method:

τ = 1.0 # HDG stabilization parameter

# ## Weak Form
#
# The HDG weak form consists of three equations coupling q, u, and λ.
# We need operators to help define the weak form:

n = get_normal_vector(Γp)  # Face normal vector
Πn(u) = u⋅n  # Normal component
Π(u) = change_domain(u,Γp,DomainStyle(u))  # Project to skeleton

# The bilinear and linear forms are:
# 1. Volume integrals for flux and primal equations
# 2. Interface integrals for hybridization
# 3. Stabilization terms with parameter τ

a((uh,qh,sh),(vh,wh,lh)) = ∫( qh⋅wh - uh*(∇⋅wh) - qh⋅∇(vh) )dΩp + ∫(sh*Πn(wh))dΓp +
                           ∫((Πn(qh) + τ*(Π(uh) - sh))*(Π(vh) + lh))dΓp
l((vh,wh,lh)) = ∫( f*vh )dΩp

# ## Static Condensation and Solution
#
# A key feature of HDG is static condensation - eliminating volume variables
# to get a smaller system for λ only:

op = MultiField.StaticCondensationOperator(ptopo,X,a,l)
uh, qh, sh = solve(op)

# ## Error Analysis
#
# Compute the L2 error between numerical and exact solutions:

dΩ = Measure(Ω,degree)
eh = uh - u
l2_uh = sqrt(sum(∫(eh⋅eh)*dΩ))

writevtk(Ω,"results",cellfields=["uh"=>uh,"qh"=>qh,"eh"=>eh])

# ## Going Further
#
# By modifying the stabilisation term, HDG can also work on polytopal meshes. An driver 
# solving the same problem on a polytopal mesh is available [in the Gridap repository](https://github.com/gridap/Gridap.jl/blob/75efc9a7a7e286c27e7ca3ddef5468e591845484/test/GridapTests/HDGPolytopalTests.jl). 
# A tutorial for HHO on polytopal meshes is also available. 
#
