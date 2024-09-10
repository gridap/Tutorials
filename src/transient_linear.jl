# In this tutorial, we will learn
# * How to solve a simple time-dependent PDE in Gridap
# * How to build a transient FE space
# * How to define a transient weak form
# * How to set up a time-marching scheme for a linear ODE
# * How to visualise transient results

# We split the presentation of the ODE module of Gridap in two parts:
# * In this tutorial we focus on the differences between the steady and time-dependent cases, on a simple linear time-dependent PDE.
# * [Tutorial 18](@ref transient_nonlinear.jl) will introduce more advanced features of the ODE module of Gridap, applied to a nonlinear time-dependent PDE.

# The [documentation of the ODE module of Gridap](https://gridap.github.io/Gridap.jl/dev/ODEs/) contains a detailed description of the framework for transient problems implemented in Gridap, including a [classification of transient problem](https://gridap.github.io/Gridap.jl/dev/ODEs/#Classification-of-ODEs), a [classification of numerical schemes](https://gridap.github.io/Gridap.jl/dev/ODEs/#Classification-of-numerical-schemes), an overview of the [high-level API](https://gridap.github.io/Gridap.jl/dev/ODEs/#High-level-API-in-Gridap) which this tutorial illustrates and of the [internals of the ODE module](https://gridap.github.io/Gridap.jl/dev/ODEs/#Low-level-implementation), as well as some [notes on and analysis of the numerical schemes implemented in Gridap](https://gridap.github.io/Gridap.jl/dev/ODEs/#Numerical-schemes-formulation-and-implementation).

# ## Problem statement

# We solve the heat equation in the two-dimensional domain $\Omega = [-1, +1]^{2}$. Let $k$ denote the thermal conductivity of the material, $c$ its specific heat capacity, $\rho$ its density and $q$ a rate of external heat generation. Let $g$ denote the temperature on the boundary of the domain. Let $t_{0}$ be the initial time, and $u_{0}$ be the initial temperature. The strong form of the heat equation reads: find $u(t): \Omega \to \mathbb{R}$ such that
# ```math
# \left\lbrace
# \begin{aligned}
# \rho(t, x) c(t, x) \partial_{t} u(t, x) - \nabla \cdot (k(t, x) \nabla u(t, x)) &= q(t, x) & \text{ in } \Omega, \\
# u(t, x) &= g(t, x) & \text{ on } \partial \Omega, \\
# u(t_{0}, x) &= u_{0}(x) & \text{ in } \Omega \\
# \end{aligned}
# \right.
# ```
# We assume that the data ($k$, $c$, $\rho$ and $q$) is continuous in time. Let $\alpha = k / (\rho c)$ denote the thermal diffusivity and $f = q / (\rho c)$ be the rate of external temperature generation. The weak form of the problem reads: find $u(t) \in U_{g}(t)$ such that $b(t, u, v) = \ell(t, v)$ for all $t \geq t_{0}$ and $v \in V_{0}$, where the time-dependent bilinear and linear forms $b(t, \cdot, \cdot)$ and $\ell(t, \cdot)$ are defined as
# ```math
# \begin{aligned}
# b(t, u, v) &= m(t, u, v) + a(t, u, v), \\
# m(t, u, v) &= \int_{\Omega} v \partial_{t} u(t) \ {\rm d} \Omega, \\
# a(t, u, v) &= \int_{\Omega} \alpha(t) \nabla v \cdot \nabla u(t) \ {\rm d} \Omega, \\
# \ell(t, v) &= \int_{\Omega} v f(t) \ {\rm d} \Omega,
# \end{aligned}
# ```
# and the the functional spaces are $U_{g}(t) = H^{1}_{g(t)}(\Omega)$ and $V_{0} = H^{1}_{0}(\Omega)$. In particular, the trial space $U_{g}$ is a transient functional space, in the sense that its Dirichlet trace $g$ depends on time. However, the test space $V_{0}$ is time-independent (it has a constant, zero Dirichlet trace). For all $t \geq t_{0}$, we assume that $\alpha(t) \in L^{\infty}(\Omega)$ is uniformly positive in $\Omega$, $f(t) \in H^{-1}(\Omega)$, $g(t) \in H^{1/2}(\Omega)$, and finally $u_{0} \in L^{2}(\Omega)$.

# ## Discrete model

# We start with the discretization of the computational domain. In our case, we consider a $20 \times 20$ Cartesian mesh of the square $[-1, +1]^{2}$.

using Gridap
domain = (-1, +1, -1, +1)
partition = (20, 20)
model = CartesianDiscreteModel(domain, partition)

# ## FE spaces

# In this tutorial we use a linear Lagrangian FE space.

order = 1
reffe = ReferenceFE(lagrangian, Float64, order)

# The test space is defined as for steady problems

V0 = TestFESpace(model, reffe, dirichlet_tags="boundary")

# The trial space is now a `TransientTrialFESpace`, which is constructed from a `TestFESpace` and a function (or vector of functions if several Dirichlet tags are provided) for the Dirichlet boundary conditions. The Dirichlet trace has to be prescribed as a function of time and then space as follows

g(t) = x -> exp(-2 * t) * sinpi(t * x[1]) * (x[2]^2 - 1)
Ug = TransientTrialFESpace(V0, g)

# ## Triangulation and quadrature

# As usual, we equip the model with an integration mesh and a measure

degree = 2
Ω = Triangulation(model)
dΩ = Measure(Ω, degree)

# ## Weak form
# We define the thermal diffusivity $\alpha$ and the rate of external temperature generation $f$.

α(t) = x -> 1 + sin(t) * (x[1]^2 + x[2]^2) / 4
f(t) = x -> sin(t) * sinpi(x[1]) * sinpi(x[2])

# We are going to construct a transient linear FEOperator by providing the bilinear forms associated to $\partial_{t} u$ and $u$, as well as the forcing term. Note that they now receive time as an additional argument, and the time derivative operator is `∂t`.

m(t, dtu, v) = ∫(v * dtu)dΩ
a(t, u, v) = ∫(α(t) * ∇(v) ⋅ ∇(u))dΩ
l(t, v) = ∫(v * f(t))dΩ
op = TransientLinearFEOperator((m, a), l, Ug, V0)

# In our case, the mass term ($m(t, \cdot, \cdot)$) is constant in time. We can take advantage of that to save some computational effort, and indicate it to Gridap as follows
op_opt = TransientLinearFEOperator((m, a), l, Ug, V0, constant_forms=(true, false))

# If the stiffness term ($a(t, \cdot, \cdot)$) had been constant in time, we could have set `constant_forms=(true, true)`.

# ## Transient solver

# Once we have defined the FE operator, we proceed with the definition of the ODE solver, i.e. the scheme that will be used for the integration in time. In this tutorial, we use the `ThetaMethod` with $\theta = 1/2$, resulting in a second-order scheme. The `ThetaMethod` function receives a solver for systems of equations, the time step size $\Delta t$ (constant) and the value of $\theta \in [0, 1]$. Since the ODE is linear the systems of equation that will arise in the time-marching scheme will be linear so we can provide `ThetaMethod` with a linear solver.

ls = LUSolver()
Δt = 0.05
θ = 0.5
solver = ThetaMethod(ls, Δt, θ)

# Gridap also implements explicit and diagonally-implicit Runge-Kutta schemes. One can access the full list of available Butcher tableaus through the exported constant `available_tableaus`. There are also constructors for explicit 2- and 3-stage schemes: `EXRK22(α)` and `EXRK33(α, β)`, `EXRK33_1(α)`, `EXRK33_2(α)` respectively, and diagonally-implicit 1- and 2-stage schemes: `SDIRK11(α)`, `SDIRK12()`, `SDIRK22(α, β, γ)`, `SDIRK23(λ)`. See the documentation of [Runge-Kutta schemes in Gridap](https://gridap.github.io/Gridap.jl/dev/ODEs/#Runge-Kutta) for a description of the corresponding tableaus. For example, one could have chosen a two-stage singly-diagonally-implicit scheme (of order 2) as follows.
tableau = :SDIRK_2_2
solver_rk = RungeKutta(ls, ls, Δt, tableau)

# Let $t_{F} > t_{0}$ be a final time, until when we want to evolve the problem. We define the solution using the `solve` function, giving the ODE solver, the transient FE operator, the initial and final times, and the initial solution. To construct the initial condition we interpolate the initial function $u_{0}$ onto the FE space $U_{g}$ at the initial time. In our case, $u_{0}$ is simply $g(t_{0})$.

t0, tF = 0.0, 10.0
uh0 = interpolate_everywhere(g(t0), Ug(t0))
uh = solve(solver, op, t0, tF, uh0)

# ## Postprocessing

# We highlight that `uh` is an iterable function and the result at each time step is only computed lazily when iterating over it. We can post-process the results and generate the corresponding `vtk` files using the `createpvd` and `createvtk` functions. The former will create a `.pvd` file with the collection of `.vtu` files saved at each time step by `createvtk`. The computation of the problem solutions will be triggered in the following loop:

if !isdir("tmp")
  mkdir("tmp")
end

createpvd("results") do pvd
  pvd[0] = createvtk(Ω, "tmp/results_0" * ".vtu", cellfields=["u" => uh0])
  for (tn, uhn) in uh
    pvd[tn] = createvtk(Ω, "tmp/results_$tn" * ".vtu", cellfields=["u" => uhn])
  end
end

# ![](../assets/transient_linear/result.gif)
