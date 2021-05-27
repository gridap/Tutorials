

#md # !!! note
#
#     This tutorial is under construction, but the code below is already functional.
#

# This tutorial builds on the steady Incompressible Navier-Stokes tutorial. Here we will learn
# the following additional feature:
#  - How to solve a transient nonlinear multi-field problem in Gridap
#
# ## Problem statement
# ### Strong form
#
# The goal of this tutorial is to solve a transient nonlinear multi-field PDE. As a model problem, we consider a well known benchmark in computational fluid dynamics, the flow around a cylinder. Formally, the PDE we want to solve is: find the velocity vector $u$ and the pressure $p$ such that
#
# ```math
# \left\lbrace
# \begin{aligned}
# \partial_t u-2\nu\epsilon(u) + (u\cdot \nabla)\ u + \nabla p = 0 &\text{ in }\Omega,\\
# \nabla\cdot u = 0 &\text{ in } \Omega,\\
# u = u_{\text{\footnotesize in}} &\text{ on } \Gamma_{\text{\footnotesize in}},\\
# u = 0 &\text{ on } \Gamma_{\text{\footnotesize wall}},\\
# (2\nu\epsilon(u)-p\mathbf{I})⋅n_{\text{\footnotesize out}} = 0 &\text{ on } \Gamma_{\text{\footnotesize out}},\\
# \end{aligned}
# \right.
# ```
#
# where $\epsilon(u)=\frac{1}{2}(\nabla u +\nabla u^T)$ is the symmetric gradient operator applied to the velocity vector.

# ### Geometry and Discrete model
# 
# The computational domain, $\Omega$, is a channel of heigh $H=0.41$ and length $L=2.0$, with a cylinder of diameter $\varnothing=0.1$ and centre coordinates $(x_c,y_c)=(0.2,0.2)$ from the left-bottom corner. 
# The left side of the channel is the inlet boundary, $\Gamma_{\text{\footnotesize in}}$, the right side of the channel is the outlet boundary, $\Gamma_{\text{\footnotesize out}}$, and the wall boundary, $\Gamma_{\text{\footnotesize wall}}$ is composed by the top and bottom sides, together with the cylinder.
#

# !!! todo
#     Add figure
#

# Before going further, let's load Gridap library so we can use some of the functionalities.

using Gridap

# The geometry for this problem has been generated using Gmsh and can be loaded by calling the `DiscreteModelFromFile` function and sending the file located at `"../models/cylinder_NSI.json"`.
model = DiscreteModelFromFile("../models/cylinder_NSI.json")

# We can inspect the loaded geometry and associated parts by printing to a `vtk` file:
writevtk(model, "model")

# ### Boundary conditions and problem parameters

#
# In this example, the driving force is given by the inlet Dirichlet boundary velocity $u_{\text{\footnotesize in}}$, which is defined by an horizontal velocity with a constant parabolic profile:
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


# The inlet condition, $u_{\text{\footnotesize in}}$, and the no-slip condition at the walls, $u_{\text{\footnotesize wall}}$, can be defined as a function of space and time as follows:

u_in(x, t::Real) = VectorValue(1.5 * Um * x[2] * (H - x[2]) / ((H / 2)^2), 0.0)
u_wall(x, t::Real) = VectorValue(0.0, 0.0)

# To evaluate the boundary conditions at certain time instances, we also need a function that, given a fixed time $t$, it returns a function of space only. That is achieved exploiting julia's multiple-dispatch feature:

u_in(t::Real) = x -> u_in(x, t)
u_wall(t::Real) = x -> u_wall(x, t)

# ## Numerical Scheme
# ### FE spaces
#
# In order to approximate this problem we chose a formulation based on inf-sub stable $P_k/P_{k-1}$ elements with continuous velocities and pressures (see, e.g., [1] for specific details). The interpolation spaces are defined as follows.  

# ##### Velocity FE space
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

# ##### Pressure FE space
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

# Finally, we glue the test and trial FE spaces together, defining a unique test and trial space for all the fields using the `MultiFieldFESpace` function. That is $Y=[V_0, Q]^T$ and $X=[U, P]^T$
Y = MultiFieldFESpace([V₀,Q])
X = MultiFieldFESpace([U,P])

# ### Numerical integration
# To define the quadrature rules used in the numerical integration of the different terms, we first need to generate the domain triangulation. Here we create the triangulation of the global domain, $\mathcal{T}$.
Ω = Triangulation(model)

# Once we have the triangulation, we can generate the quadrature rules. This will be generated by calling the `Measure` function, that given a triangulation and an integration degree, it returns the Lebesgue integral measure $d\Omega$.
degree = 2*k
dΩ = Measure(Ω,degree)

# ### Weak form
# The weak form of the transient Navier-Stokes problem reads: find $[\mathbf{u}^h, p^h]^T \in X$ such that
# ```math
# a([\mathbf{u}^h, p^h],[\mathbf{v}^h, q^h])=l([\mathbf{v}^h, q^h])\qquad\forall[\mathbf{v}^h, q^h]^T\in Y,
# ```
# where
# ```math
# \begin{aligned}
# &a([\mathbf{u}^h, p^h],[\mathbf{v}^h, q^h])\doteq \int_{\Omega} \left[ \mathbf{v}\cdot\partial_t\mathbf{u}+ \mathbf{v}\cdot(\mathbf{u}\cdot\nabla)\mathbf{u} + 2\nu\epsilon(\mathbf{v}):\epsilon(\mathbf{u}) - (\nabla\cdot \mathbf{v}) \ p + q \ (\nabla \cdot \mathbf{u})\right] \ {\rm d}\Omega,\\
# &l([\mathbf{v}^h, q^h])\doteq 0.
# \end{aligned}
# ```
a((u,ut,p),(v,q)) = ∫( v⋅ut + v⋅((∇(u)')⋅u) + 2*ν*(ε(v)⊙ε(u)) - (∇⋅v)*p + q*(∇⋅u) )dΩ
l((v,q)) = 0

# Note that we keep the velocity time derivative as a variable. The main reason behind this approach is that in this way, the variational form is not tied to a particular time integrator. This allows to have a time discretization that is handled internally by `GridapODEs`. 

# Due to the presence of the convective term, this problem is nonlinear. To solve a time-dependent nonlinear problem we define a transient nonlinear FE operator from the residual (`res`), jacobian with respect to the unknowns (`jac`) and jacobian with respect to the unknowns' time derivative (`jac_t`). The `TransientFEOperator` expects a residual function with three arguments: the time `t`, a Tuple with the unknowns and unknowns' time derivative `((u,p),(ut,))` and the test functions `(v,q)`. 
res(t,((u,p),(ut,)),(v,q)) = a((u,ut,p),(v,q)) - l((v,q))

# The Jacobian with respect to the unknowns expects: the time `t`, a Tuple with the unknowns and unknowns' time derivative `((u,p),(ut,))`, i.e. linearization point, the linearized unknowns `(du,dp)` and the test functions `(v,q)`. 
jac(t,((u,p),(ut,)),(du,dp),(v,q)) = ∫( v⋅((∇(du)')⋅u) + v⋅((∇(u)')⋅du) + 2*ν*(ε(v)⊙ε(du)) - (∇⋅v)*dp + q*(∇⋅du) )dΩ

# Finally, the Jacobian with respect to the unknowns' time derivative expects: the time `t`, a Tuple with the unknowns and unknowns' time derivative `((u,p),(ut,))`, i.e. linearization point, the linearized unknowns' time derivative `(dut,)` and the test functions `(v,q)`.
jac_t(t,((u,p),(ut,)),(dut,),(v,q)) = ∫( v⋅dut )dΩ

# With the residuals and jacobians defined, we can construct the FE operator.
op = TransientFEOperator(res,jac,jac_t,X,Y)

# One of the features of Gridap and GridapODEs is the ability to automatically derive the jacobians with respect to the unknowns and unknowns' time derivative, using automatic differentiation tools provided by `ForwardDiff`. This is achieved by simply calling the `TransientFEOperator` with the residual and the variational spaces as the only arguments.
using ForwardDiff
op_AD = TransientFEOperator(res,X,Y)

#
# The weak form associated to these interpolation spaces reads: find $(u,p)\in U_g \times Q_0$ such that $[r(u,p)](v,q)=0$ for all $(v,q)\in V_0 \times Q_0$
# where $U_g$ and $V_0$ are the set of functions in $V$ fulfilling the Dirichlet boundary condition $g$ and $0$  on $\partial\Omega$ respectively. The weak residual $r$ evaluated at a given pair $(u,p)$ is the linear form defined as
#
# ```math
# [r(u,p)](v,q) \doteq a((u,p),(v,q))+ [c(u)](v),
# ```
# with
# ```math
# \begin{aligned}
# a((u,p),(v,q)) &\doteq \int_{\Omega} \nabla v \cdot \nabla u \ {\rm d}\Omega - \int_{\Omega} (\nabla\cdot v) \ p \ {\rm d}\Omega + \int_{\Omega} q \ (\nabla \cdot u) \ {\rm d}\Omega,\\
# [c(u)](v) &\doteq \int_{\Omega} v 	\cdot \left( (u\cdot\nabla)\ u \right)\ {\rm d}\Omega.\\
# \end{aligned}
# ```
# Note that the bilinear form $a$ is associated with the linear part of the PDE, whereas $c$ is the contribution to the residual resulting from the convective term.
#

using LinearAlgebra
using GridapODEs.ODETools
using Gridap.FESpaces: get_algebraic_operator
using WriteVTK
using LineSearches: BackTracking
using Plots
import GridapODEs.TransientFETools: ∂t

# ## Problem setting
# Parameters
const t0 = 0.0

# Boundary conditions

∂tu_in(t) = x -> VectorValue(0.0, 0.0)
∂tu_in(x, t) = ∂tu_in(t)(x)
∂t(::typeof(u_in)) = ∂tu_in
∂t(::typeof(u_noSlip)) = ∂tu_in

# ## Domain
# Toy model to initialize julia
n = 2
domain = (0, 1, 0, 1)
partition = (n, n)
model0 = CartesianDiscreteModel(domain, partition)
labels0 = get_face_labeling(model0)
add_tag_from_tags!(labels0, "inlet", [5])
add_tag_from_tags!(labels0, "noslip", [6])
add_tag_from_tags!(labels0, "cylinder", [7])
add_tag_from_tags!(labels0, "outlet", [8])

# Model from GMSH
labels = get_face_labeling(model)


# ## Weak form
# Laws
@law conv(u, ∇u) = (∇u') ⋅ u
@law dconv(du, ∇du, u, ∇u) = conv(u, ∇du) + conv(du, ∇u)
@law σ_dev(ε) = 2 * ν * ε

# Bilinear function
function a(x, y)
    u, p = x
    v, q = y
    (ε(v) ⊙ σ_dev(ε(u))) - (∇⋅v) * p + q * (∇⋅u)
end

# Nonlinear convective term
c(u, v) = v ⊙ conv(u, ∇(u))
dc(u, du, v) = v ⊙ dconv(du, ∇(du), u, ∇(u))

# Navier-Stokes residual
function res(t, x, xt, y)
    u, p = x
    ut, pt = xt
    v, q = y
    (ut ⊙ v) + a(x, y) + c(u, v)
end

# Navier-Stokes jacobian (w.r.t space)
function jac(t, x, xt, dx, y)
    u, p = x
    v, q = y
    du, dp = dx
    a(dx, y) + dc(u, du, v)
end

# Navier-Stokes jacobian (w.r.t. time)
function jac_t(t, x, xt, dxt, y)
    dut, dpt = dxt
    v, q = y
    (dut ⊙ v)
end

# ## Solver functions
# Stokes
function solveStokes(op)
    ls = LUSolver()
    solver = FESolver(ls)
    xh0 = solve(solver, op)
end
# Navier-Stokes
function solveNavierStokes(op, xh0, t0, tF, dt, θ)
    nls = NLSolver(
        show_trace = false,
        method = :newton,
        linesearch = BackTracking(),
    )
    odes = ThetaMethod(nls, dt, θ)
    solver = TransientFESolver(odes)
    sol_t = solve(solver, op, xh0, t0, tF)
end

# ## PostProcessing
# Write to vtk
function writePVD(filePath::String, trian::Triangulation, sol; append=false)
    outfiles = paraview_collection(filePath, append=append) do pvd
        for (i, (xh, t)) in enumerate(sol)
            uh = xh.blocks[1]
            ph = xh.blocks[2]
            pvd[t] = createvtk(
                trian,
                filePath * "_$t.vtu",
                cellfields = ["uh" => uh, "ph" => ph],
            )
        end
    end
end

# Compute forces
function computeForces(model::DiscreteModel, sol, xh0)

    ## Surface triangulation
    trian_Γc = BoundaryTriangulation(model, "cylinder")
    quad_Γc = CellQuadrature(trian_Γc, 2)
    n_Γc = get_normal_vector(trian_Γc)

    ## Drag & Lift coefficients
    coeff(F) = 2 * F / (ρ * Um^2 * ∅)

    ## Initialize arrays
    tpl = Real[]
    CDpl = Real[]
    CLpl = Real[]
    uhn = xh0[1]
    phn = xh0[2]

    ## Get solution at n+θ (where pressure and velocity are balanced)
    θ = 0.5

    ## Loop over steps
    for (xh, t) in sol

        ## Get the solution at n+θ (where velocity and pressure are balanced)
        uh = xh.blocks[1]
        ph = xh.blocks[2]
        phθ = θ * ph + (1.0 - θ) * phn
        uh_Γc = restrict(uh, trian_Γc)
        uhn_Γc = restrict(uhn, trian_Γc)
        ph_Γc = restrict(phθ, trian_Γc)
        εθ = θ * ε(uh_Γc) + (1.0 - θ) * ε(uhn_Γc)
        FD, FL = sum(integrate(
            (n_Γc ⋅ σ_dev(ε(uh_Γc))  - ph_Γc * n_Γc),
            trian_Γc,
            quad_Γc,
        ))

        ## Drag and lift coefficients
        push!(tpl, t)
        push!(CDpl, -coeff(FD))
        push!(CLpl, coeff(FL))

        ## store step n
        uhn = uh
        phn = ph
    end

    return (tpl, CDpl, CLpl)
end


# ## Simutation function
function runCylinder(model::DiscreteModel, labels)

    ### FE spaces
    ## Test FE spaces
    D = 2
    order = 2

    ## Trial FE spaces
    U = TransientTrialFESpace(V, [u_in, u_noSlip, u_noSlip])
    U0 = TrialFESpace(V, [u_in(t0), u_noSlip(t0), u_noSlip(t0)])
    P = TrialFESpace(Q)

    ## Multifield FE spaces
    Y = MultiFieldFESpace([V, Q])
    X = MultiFieldFESpace([U, P])
    X0 = MultiFieldFESpace([U0, P])

    ### Triangulation and CellQuadrature
    trian = Triangulation(model)
    degree = (order - 1) * 2
    quad = CellQuadrature(trian, degree)

    ### FE operators
    ## Navier-Stokes FE operator
    t_Ω = FETerm(res, jac, jac_t, trian, quad)
    op = TransientFEOperator(X, Y, t_Ω)
    ## Stokes FE operator
    t_Stokes_Ω = LinearFETerm(a, trian, quad)
    op_Stokes = FEOperator(X0, Y, t_Stokes_Ω)

    ### Stokes solution
    println("solveStokes")
    xh0 = solveStokes(op_Stokes)

    ### Initialize Paraview files
    folderName = "ins-results"
    fileName = "fields"
    if !isdir(folderName)
        mkdir(folderName)
    end
    filePath = join([folderName, fileName], "/")

    ### Output to Paraview
    println("writeStokes")
    writePVD(filePath, trian, [(xh0, 0.0)])

    ### Transient Navier-Stokes solution (initial stage to reach fully developed flow)
    println("solveNavierStokes 1")
    sol_t = solveNavierStokes(op, xh0, 0.0, 8.0, 0.05, 0.5)

    ### Output transient solution to Paraview
    println("writeNavierStokes 1")
    writePVD(filePath, trian, sol_t, append=true)

    ### Get last solution snapshot as initial condition (this should be done overwriting Base.last)
    for (xh_tn, tn) in sol_t
        xh0 = xh_tn
    end

    ### Transient Navier Stokes solution to pick statistics
    println("solveNavierStokes 2")
    sol_t = solveNavierStokes(op, xh0, 8.0, 10.0, 0.005, 0.5)

    ### Output transient solution to Paraview
    println("writeNavierStokes 2")
    writePVD(filePath, trian, sol_t, append=true)

    ### Output drag and lift coefficients
    (t, CD, CL) = computeForces(model, sol_t, xh0)
    return (t, CD, CL)

end

# ## Execute simulation
#(t, CD, CL) = @time runCylinder(model0, labels0)
(t, CD, CL) = @time runCylinder(model, labels)
p1 = plot(t, CD, label="CD")
p2 = plot(t, CL, label="CL")
display(plot(p1,p2,layout=(1,2)))
println("CDmax: ", maximum(CD), ", CLmax: ", maximum(CL))

# ## Results

# Velocity field animation
# ![](../assets/cylinder_ins/cylinder.gif)

# Drag and lift coefficients evolution
# ![](../assets/cylinder_ins/cylinder_coeff.png)

#
#
# | QoI | CD_max | CL_max |
# | :---: | :---:| :---: |
# | Computed | 3.178 | 1.012 |
# | Reference range | [3.220, 3.24] | [0.990,1.010] |
#

# ## References
# [1] Schäfer, Michael, et al. *Benchmark computations of laminar flow around a cylinder.* Flow simulation with high-performance computers II. Vieweg+ Teubner Verlag, 1996. 547-566.
