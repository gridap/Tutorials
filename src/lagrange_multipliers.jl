# In this tutorial, we will learn
#
#    - How to enforce constraints using Lagrange multipliers
#    - How to work with `ConstantFESpace`
#
# ## Problem statement
#
# In this tutorial, we solve the Poisson equation with pure Neumann boundary conditions. 
# This problem is well-known to be singular since the solution is defined up to a constant, 
# which we have to fix to obtain a unique solution.
# Here, we will use a Lagrange multiplier to enforce that the mean value of the solution 
# equals a given constant.
#
# The problem reads: find $u$ and $λ$ such that
#
# ```math
# \left\lbrace
# \begin{aligned}
# -\Delta u = f  \ &\text{in} \ \Omega,\\
# \nabla u\cdot n = g \ &\text{on}\ \Gamma,\\
# \int_{\Omega} u \ {\rm d}\Omega = \bar{u},\\
# \end{aligned}
# \right.
# ```
#
# where $\Omega$ is our domain, $\Gamma$ is its boundary, $n$ is the outward unit normal vector,
# and $\bar{u}$ is a given constant that fixes the mean value of the solution.
#
# ## Numerical scheme
#
# The weak form of this problem using Lagrange multipliers reads:
# find $(u,λ) \in V \times \Lambda$ such that
#
# ```math
# \begin{aligned}
# \int_{\Omega} \nabla u \cdot \nabla v \ {\rm d}\Omega + 
# \int_{\Omega} λv \ {\rm d}\Omega + 
# \int_{\Omega} uμ \ {\rm d}\Omega = 
# \int_{\Omega} fv \ {\rm d}\Omega + 
# \int_{\Gamma} v(g\cdot n) \ {\rm d}\Gamma + 
# \int_{\Omega} μ\bar{u} \ {\rm d}\Omega
# \end{aligned}
# ```
#
# for all $(v,μ) \in V \times \Lambda$, where $V = H^1(\Omega)$ and $\Lambda = \mathbb{R}$.
#
# ## Implementation
#
# First, we load the Gridap package and define the exact solution that we will use to 
# manufacture the source term and boundary condition:

using Gridap

u_exact(x) = sin(x[1]) * cos(x[2])

# Now we can create a simple Cartesian mesh of the unit square:

model = CartesianDiscreteModel((0,1,0,1),(8,8))

# We will use first order Lagrangian finite elements for the primal variable u.

order = 1
reffe = ReferenceFE(lagrangian, Float64, order)
V = FESpace(model, reffe)

# For the Lagrange multiplier λ, we need a space of constant functions, since λ ∈ ℝ.
# In Gridap, we can create such a space using `ConstantFESpace`:

Λ = ConstantFESpace(model)

# Conceptually, a `ConstantFESpace` is a space defined on the whole domain with a 
# single degree of freedom, which is what we need for the Lagrange multiplier λ.
# We finally bundle both spaces into a multi-field space:

X = MultiFieldFESpace([V, Λ])

# ## Integration
#
# We need to create the triangulation and measures for both domain and boundary 
# integration:

Ω = Triangulation(model)
Γ = BoundaryTriangulation(model)
dΩ = Measure(Ω, 2*order)
dΓ = Measure(Γ, 2*order)

# Next, we manufacture the source term f and Neumann boundary condition g 
# from the exact solution. We also compute the mean value ū that we want 
# to enforce:

f(x) = -Δ(u_exact)(x)
g(x) = ∇(u_exact)(x)
ū = sum(∫(u_exact)dΩ)
nΓ = get_normal_vector(Γ)

# ## Weak Form
#
# We can now define the bilinear and linear forms of our problem.
# Note how the forms take tuples as arguments, representing the 
# multi-field nature of our solution:

a((u,λ),(v,μ)) = ∫(∇(u)⋅∇(v) + λ*v + u*μ)dΩ
l((v,μ)) = ∫(f*v + μ*ū)dΩ + ∫(v*(g⋅nΓ))*dΓ

# ## Solution
#
# We can now create the FE operator and solve the system:

op = AffineFEOperator(a, l, X, X)
uh, λh = solve(op)

# Note how we get two values from solve: the primal solution uh and 
# the Lagrange multiplier λh. Finally, we compute the L2 error and 
# verify that the mean value constraint is satisfied:

eh = uh - u_exact
l2_error = sqrt(sum(∫(eh⋅eh)*dΩ))
ūh = sum(∫(uh)*dΩ)

# The L2 error should be small (of order h²) and ūh should be very close to ū,
# showing that both the equation and the constraint are well satisfied.

# ## Visualization
#
# We can visualize the solution and error by writing them to a VTK file:

mkpath("output_path")
writevtk(Ω, "output_path/results", cellfields=["uh"=>uh, "error"=>eh])
