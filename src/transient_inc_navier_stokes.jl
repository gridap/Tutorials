# This tutorial builds on the steady Incompressible Navier-Stokes tutorial. Here we will learn
# the following additional features:
#  - How to solve a transient nonlinear multi-field problem in Gridap.
#
# 1. [Problem Statement](#probStat)
#    - [Strong form](#strongForm)
#    - [Geometry and Discrete model](#geometry)
#    - [Boundary conditions and problem parameters](#conditions)
# 2. [Numerical scheme](#numericalScheme)
#    - [FE spaces](#feSpace)
#    - [Numerical integration](#integration)
#    - [Weak form](#weakForm)
#    - [Algebraic system of equations](#algebraic)
# 3. [FE solution] (#feSolution)
#    - [Nonlinear solver](#nonlinearSolver)
#    - [Time integrator](#timeIntegrator)
#    - [Transient FE solver](#transientFESolver)
#    - [Transient FE solution](#transientFESolution)
# 4. [Post-processing](#postprocess)
#    - [Force coefficients](#force)
# 5. [Results](#results)
#
# ## Problem statement
# #### Strong form
#
# The goal of this tutorial is to solve a transient nonlinear multi-field PDE. As a model problem, we consider a well known benchmark in computational fluid dynamics, the flow around a cylinder. Formally, the PDE we want to solve is: find the velocity vector $u$ and the pressure $p$ such that
#
# ```math
# \left\lbrace
# \begin{aligned}
# \partial_t u-2\nu\epsilon(u) + (u\cdot \nabla)\ u + \nabla p = \mathbf{f} &\text{ in }\Omega,\\
# \nabla\cdot u = 0 &\text{ in } \Omega,\\
# u = u_{\text{\footnotesize in}} &\text{ on } \Gamma_{\text{\footnotesize in}},\\
# u = 0 &\text{ on } \Gamma_{\text{\footnotesize wall}},\\
# (2\nu\epsilon(u)-p\mathbf{I})⋅n_{\text{\footnotesize out}} = 0 &\text{ on } \Gamma_{\text{\footnotesize out}},\\
# \end{aligned}
# \right.
# ```
#
# where $\epsilon(u)=\frac{1}{2}(\nabla u +\nabla u^T)$ is the symmetric gradient operator applied to the velocity vector.

# #### Geometry and Discrete model
#
# The computational domain, $\Omega$, is a channel of heigh $H=0.41$ and length $L=2.0$, with a cylinder of diameter $\varnothing=0.1$ and centre coordinates $(x_c,y_c)=(0.2,0.2)$ from the left-bottom corner.
# The left side of the channel is the inlet boundary, $\Gamma_{\text{\footnotesize in}}$, the right side of the channel is the outlet boundary, $\Gamma_{\text{\footnotesize out}}$, and the wall boundary, $\Gamma_{\text{\footnotesize wall}}$ is composed by the top and bottom sides, together with the cylinder.
#
# ![](../assets/cylinder_ins/cylinder_geometry.png)

# Before going further, let's load Gridap library so we can use some of the functionalities.

using Gridap

# The geometry for this problem has been generated using Gmsh and can be loaded by calling the `DiscreteModelFromFile` function and sending the file located at `"../models/cylinder_NSI.json"`.
model = DiscreteModelFromFile("../models/cylinder_NSI.json")

# We can inspect the loaded geometry and associated parts by printing to a `vtk` file:
writevtk(model, "model")

# #### Boundary conditions and problem parameters

#
# In this example, the source term is zero, i.e. $\mathbf{f}=0$, and the driving force is given by the inlet Dirichlet boundary velocity $u_{\text{\footnotesize in}}$, which is defined by an horizontal velocity with a constant parabolic profile:
#
# ```math
# u_{\text{\footnotesize in}}(x,y,t) = 1.5 U_m \frac{y (H - y)}{(H / 2)^2}.
# ```
#
# Here $U_m$ is the mean flow velocity set to $U_m=1.0$. In this tutorial we select the benchmark case with a characteristic Reynolds number of $Re=100$, resulting in a viscosity value of $\nu=\frac{U_m\varnothing}{Re}$. All the problem parameters are set as constants in the tutorial.

const Um = 1.0
const H = 0.41
const ∅ = 0.1
const Re = 100.0
const ν = Um * ∅ / Re
const f = VectorValue(0.0,0.0)


# The inlet condition, $u_{\text{\footnotesize in}}$, and the no-slip condition at the walls, $u_{\text{\footnotesize wall}}$, can be defined as a function of space and time as follows:

u_in(x, t::Real) = VectorValue(1.5 * Um * x[2] * (H - x[2]) / ((H / 2)^2), 0.0)
u_wall(x, t::Real) = VectorValue(0.0, 0.0)

# To evaluate the boundary conditions at certain time instances, we also need a function that, given a fixed time $t$, it returns a function of space only. That is achieved exploiting julia's multiple-dispatch feature:

u_in(t::Real) = x -> u_in(x, t)
u_wall(t::Real) = x -> u_wall(x, t)

# ## Numerical Scheme
# #### FE spaces
#
# In order to approximate this problem we chose a formulation based on inf-sub stable $P_k/P_{k-1}$ elements with continuous velocities and pressures (see, e.g., [1] for specific details). The interpolation spaces are defined as follows.

# ###### Velocity FE space
# The velocity interpolation space is
#
# ```math
# V \doteq \{ v \in [C^0(\Omega)]^d:\ v|_T\in [P_k(T)]^d \text{ for all } T\in\mathcal{T} \},
# ```
# where $T$ denotes an arbitrary cell of the FE mesh $\mathcal{T}$, and $P_k(T)$ is the usual continuous vector-valued Lagrangian FE space of order $k$ defined on a mesh of triangles or tetrahedra. In this tutorial we will enforce the Dirichlet boundary conditions strongly, therefore the velocity test and trial spaces are given by
#
# ```math
# \begin{aligned}
# V_0 &\doteq \{ v \in V:\ v\vert_{\Gamma_{\text{\footnotesize in}}\cup\Gamma_{\text{\footnotesize wall}}}=0 \},\\
# U &\doteq \{ v \in V:\ v\vert_{\Gamma_{\text{\footnotesize in}}}=u_{\text{\footnotesize in}},\ v\vert_{\Gamma_{\text{\footnotesize wall}}}=u_{\text{\footnotesize wall}} \}.
# \end{aligned}
# ```

# After these definitions we are ready to define the velocity FE spaces. We start by defining the reference FE for the velocity field, which is defined by a 2-dimensional `VectorValue` type `lagrangian` reference FE element of order `k` (in that case $k=2$).
const k = 2
reffeᵤ = ReferenceFE(lagrangian,VectorValue{2,Float64},k)

# Now we have all the ingredients to define the velocity FE spaces. The test space $V_0$ is created by calling `TestFESpace` and sending the `model` containing the discretization information, `reffeᵤ` with the description of the reference FE space, `:H1` conformity and the set of labels for the Dirichlet boundaries.
V₀ = TestFESpace(model, reffeᵤ, conformity=:H1, dirichlet_tags=["inlet", "noslip", "cylinder"])

# The trial velocity FE space is constructed from the test space and sending the function to be enforced in the different Dirichlet boundaries. It is important to note that, since the problem is transient, the trial FE space is a `TransientTrialFESpace`. This type accomodates the fact that the Dirichlet boundary conditions can depend on time. In order to use the `TransientTrialFESpace` we first need to load the `TransientFETools` module of the `GridapODEs` package.
using GridapODEs.TransientFETools
U = TransientTrialFESpace(V₀, [u_in, u_wall, u_wall])

# ###### Pressure FE space
#
# On the other hand, the FE space for the pressure is given by
#
# ```math
# \begin{aligned}
# Q &\doteq \{ q \in C^0(\Omega):\ q|_T\in P_{k-1}(T) \text{ for all } T\in\mathcal{T}\},
# \end{aligned}
# ```
# where $P_{k-1}(T)$ is the polynomial space of multi-variate polynomials in $T$ of degree less or equal to $k-1$. Again, here we first define the reference FE for the pressure, which is given by a scalar value type `:Lagrangian` reference FE element of order `k-1`.
reffe_p = ReferenceFE(lagrangian,Float64,k-1)

# The test and trial spaces for the pressure field are regular static spaces constructed with the `TestFESpace` and `TrialFESpace` and the corresponding reference FE element.
Q = TestFESpace(model, reffe_p, conformity=:C0)
P = TrialFESpace(Q)

# Finally, we glue the test and trial FE spaces together, defining a unique test and trial space for all the fields using the `MultiFieldFESpace` and `TransientMultiFieldFESpace` functions. That is $Y=[V_0, Q]^T$ and $X=[U, P]^T$
Y = MultiFieldFESpace([V₀,Q])
X = TransientMultiFieldFESpace([U,P])

# #### Numerical integration
# To define the quadrature rules used in the numerical integration of the different terms, we first need to generate the domain triangulation. Here we create the triangulation of the global domain, $\mathcal{T}$.
Ω = Triangulation(model)

# Once we have the triangulation, we can generate the quadrature rules. This will be generated by calling the `Measure` function, that given a triangulation and an integration degree, it returns the Lebesgue integral measure $d\Omega$.
degree = 2*k
dΩ = Measure(Ω,degree)

# #### Weak form
# The weak form of the transient Navier-Stokes problem reads: find $[\mathbf{u}^h, p^h]^T \in X$ such that
# ```math
# a([\mathbf{u}^h, p^h],[\mathbf{v}^h, q^h])=l([\mathbf{v}^h, q^h])\qquad\forall[\mathbf{v}^h, q^h]^T\in Y,
# ```
# where
# ```math
# \begin{aligned}
# &a([\mathbf{u}^h, p^h],[\mathbf{v}^h, q^h])\doteq \int_{\Omega} \left[ \mathbf{v}\cdot\partial_t\mathbf{u}+ \mathbf{v}\cdot(\mathbf{u}\cdot\nabla)\mathbf{u} + 2\nu\epsilon(\mathbf{v}):\epsilon(\mathbf{u}) - (\nabla\cdot \mathbf{v}) \ p + q \ (\nabla \cdot \mathbf{u})\right] \ {\rm d}\Omega,\\
# &l([\mathbf{v}^h, q^h])\doteq \int_{\Omega}\mathbf{v}\cdot\mathbf{f}\ d\Omega.
# \end{aligned}
# ```
a((u,ut,p),(v,q)) = ∫( v⋅ut + v⋅((∇(u)')⋅u) + 2*ν*(ε(v)⊙ε(u)) - (∇⋅v)*p + q*(∇⋅u) )dΩ
l((v,q)) = ∫(v⋅f)dΩ

# Note that we keep the velocity time derivative as a variable. The main reason behind this approach is that in this way, the variational form is not tied to a particular time integrator. This allows to have a time discretization that is handled internally by `GridapODEs`.

# Due to the presence of the convective term, this problem is nonlinear. To solve a time-dependent nonlinear problem we define a transient nonlinear FE operator from the residual (`res`), jacobian with respect to the unknowns (`jac`) and jacobian with respect to the unknowns' time derivative (`jac_t`). The `TransientFEOperator` expects a residual function with three arguments: the time `t`, a Tuple with the unknowns and unknowns' time derivative `((u,p),(ut,))` and the test functions `(v,q)`.
res(t,((u,p),(ut,)),(v,q)) = a((u,ut,p),(v,q)) - l((v,q))

# The Jacobian with respect to the unknowns expects: the time `t`, a Tuple with the unknowns and unknowns' time derivative `((u,p),(ut,))`, i.e. linearization point, the linearized unknowns `(du,dp)` and the test functions `(v,q)`.
jac(t,((u,p),(ut,)),(du,dp),(v,q)) = ∫( v⋅((∇(du)')⋅u) + v⋅((∇(u)')⋅du) + 2*ν*(ε(v)⊙ε(du)) - (∇⋅v)*dp + q*(∇⋅du) )dΩ

# Finally, the Jacobian with respect to the unknowns' time derivative expects: the time `t`, a Tuple with the unknowns and unknowns' time derivative `((u,p),(ut,))`, i.e. linearization point, the linearized unknowns' time derivative `(dut,)` and the test functions `(v,q)`.
jac_t(t,((u,p),(ut,)),(dut,),(v,q)) = ∫( v⋅dut )dΩ

# With the residuals and jacobians defined, we can construct the FE operator.
op = TransientFEOperator(res,jac,jac_t,X,Y)

# ###### Stokes operator
# In addition to the transient Navier-Stokes operator, we will also define a Stokes FE operator the solution of which ($x_h^0$) will be used as an initial condition of the problem (see Stokes tutorial for further details). By using the solution to the Stokes problem as initial condition, we make sure that the initial condition satisfies the boundary conditions and the incompressibility constraint.
res_stokes((u,p),(v,q)) = ∫( 2*ν*(ε(v)⊙ε(u)) - (∇⋅v)*p + q*(∇⋅u) )dΩ
jac_stokes((u,p),(du,dp),(v,q)) = res_stokes((du,dp),(v,q))
op_stokes = FEOperator(res_stokes,jac_stokes,X(0.0),Y(0.0))
global xh₀ = solve(op_stokes)

# ## FE solution
# Once defined the FE operator, we need to construct the solver, which will require information about the time integration scheme, the nonlinear solution method and the linear solver strategy.

# #### Nonlinear solver
# We start by defining the nonlinear solver (see p-laplacian tutorial for a detailed description of this step).
using LineSearches: BackTracking
nl_solver = NLSolver(show_trace = false,method = :newton,linesearch = BackTracking())

# #### Time integrator
# Once we have the nonlinear solver defined, we construct the time integrator scheme to be used to solve the ODE resulting after discretizing in space. In this case we use the $\theta$-method, defined in  the `ODETools` module of `GridapODEs`. Here we use $\theta=0.5$ and a constant time step size of $\Delta t = 0.05$. Note that the case $\theta=0.5$ can also be called through the function `MidPoint` and the case $\theta=1.0$ through the function `BackwardEuler`. Other implemented ODE solvers are: `RungeKutta`, based on the Runge-Kutta method with implemented schemes up to 3rd order, and `Newmark`, that implements the Newmark-beta method for 2nd order ODEs.
using GridapODEs.ODETools
const θ = 0.5
Δt₁= 0.05
ode_scheme_1 = ThetaMethod(nl_solver, Δt₁, θ)

# #### Transient FE solver
# With the information of the ODE discretization scheme, we can define the transient FE solver by calling the `TransientFESolver` function of `GridapODEs`.
solver_1 = TransientFESolver(ode_scheme_1)

# #### Transient FE solution
# Finally, the FE solution is constructed by calling the function `solve` with the transient solver, the FE operator, the initial solution, initial time and final time. In this tutorial we consider two different stages:
#
# 1. An initial stage to let the flow develop, from $t_0=0.0$ to $T=8.0$, with the Stokes solution as initial solution and time step size $\Delta t=0.05$.
# 2. A second stage to compute statistics, from $t_0=8.0$ to $T=10.0$, with the last step from the initial stage as initial solution and time step size $\Delta t=0.005$.
t₀ = 0.0
T = 8.0
xh_1 = solve(solver_1, op, xh₀, t₀, T)

# We get last solution step from $xh_1$ as initial condition (this will be done overwriting Base.last in a near future)
for (xh_tn, tn) in xh_1
    global xh₀
    xh₀ = xh_tn
end

# We redefine the FE solver for the second stage.
t₀ = 8.0
T = 10.0
Δt₂= 0.005
ode_scheme_2 = ThetaMethod(nl_solver, Δt₂, θ)
solver_2 = TransientFESolver(ode_scheme_2)
xh_2 = solve(solver_2, op, xh₀, t₀, T)

# ## PostProcessing
#
# #### Force coefficients
# We start the post-processing by defining a function that, given the solution at a given time, it returns the drag and lift coefficients. The coefficients are defined by the following function
# ```math
# C = \frac{2F}{ρ U_m^2 \varnothing}
# ```
coeff(F) = 2 * F / (Um^2 * ∅)

# Where the drag and lift forces, $F_d$ and $F_l$, respectively, are the horizontal and vertical components of the resulting force acting on the cylinder. That is, the integral of the normal traction over the cylinder.
# ```math
# \left[
# \begin{aligned}
# F_d\\
# F_l
# \end{aligned}\right] = \int_{\Gamma_c}\left(\mathbf{n}_{\Gamma_c}\cdot(2\mu\epsilon(u_h)) - p_h\mathbf{I})\right)d\Gamma.
# ```

# Let's first define the boundary triangulation around the cylinder, $\Gamma_c$, the normal to the boundary, $\mathbf{n}_{\Gamma_c}$, and the integration measure $d\Gamma_c$
Γc = BoundaryTriangulation(model, tags="cylinder")
dΓc = Measure(Γc, k)
nΓc = get_normal_vector(Γc)

# Given a solution vector $x_h$ at a given time, the coefficients computation function can be defined as
function compute_coefficients(xh)
    uh, ph = xh
    F_drag, F_lift = ∑( ∫( nΓc⋅(2*ν*ε(uh)) - ph*nΓc )dΓc )
    C_d = coeff(F_drag)
    C_l = coeff(F_lift)
    return C_d, C_l
end

# We also define three auxiliar arrays of Reals to store the history of coefficients in time.
ts = Real[]
CDs = Real[]
CLs = Real[]

# We encapsulate the post-processing of the results in a `do` block in which a .pvd file is created (Paraview file with information of the solution at different time steps). This is done by calling the
# `paraview_collection` function of the `WriteVTK` package. Note that we compute the coefficients at $t^{n+\theta}$, where velocities and pressure are in equilibrium.
using WriteVTK
filePath = "results"
output_files = paraview_collection(filePath, append=false) do pvd
    for (i, (xh, t)) in enumerate(xh_2)
        global xh₀

        uh, ph = xh
        CD, CL = θ.*compute_coefficients(xh) .+ (1-θ).*compute_coefficients(xh₀)
        push!(ts,t)
        push!(CDs,-CD)
        push!(CLs,CL)

        pvd[t] = createvtk(Ω, filePath * "_$t.vtu", cellfields = ["uh" => uh, "ph" => ph])

        xh₀ = interpolate_everywhere(xh,X(t))
    end
end


# ## Results

# The resulting velocity field is given in the following animation
# ![](../assets/cylinder_ins/cylinder.gif)

# The drag and lift coefficients evolution is also reported in the following figures, obtained using `Plots`.
using Plots
p1 = plot(ts, CDs, label="CD")
p2 = plot(ts, CLs, label="CL")
plt = plot(p1,p2,layout=(1,2))
display(plt)
# `savefig(plt,"../assets/cylinder_ins/cylinder_coeff.png")`
# ![](../assets/cylinder_ins/cylinder_coeff.png)

# One can also compute the maximum drag and lift and compare it to the reference results.
println("CDmax: ", maximum(CDs), ", CLmax: ", maximum(CLs))
# | QoI | CD_max | CL_max |
# | :---: | :---:| :---: |
# | Computed | 3.182 | 1.011 |
# | Reference range | [3.220, 3.24] | [0.990,1.010] |
#

# ## References
# [1] Schäfer, Michael, et al. *Benchmark computations of laminar flow around a cylinder.* Flow simulation with high-performance computers II. Vieweg+ Teubner Verlag, 1996. 547-566.
