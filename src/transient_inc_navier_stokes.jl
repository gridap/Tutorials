
#md # !!! note
#
#     This tutorial is under construction, but the code below is already functional.
#

using Gridap
using LinearAlgebra
using GridapODEs.ODETools
using GridapODEs.TransientFETools
using Gridap.FESpaces: get_algebraic_operator
using WriteVTK
using LineSearches: BackTracking
using Plots
import GridapODEs.TransientFETools: ∂t

## Problem setting
# Parameters
const Um = 1.0
const H = 0.41
const ⌀ = 0.1
const ρ = 1.0
const t0 = 0.0
const Re = 100.0
const ν = Um * ⌀ / Re

# Boundary conditions
u_in(x, t) = VectorValue(1.5 * Um * x[2] * (H - x[2]) / ((H / 2)^2), 0.0)
u_noSlip(x, t) = VectorValue(0.0, 0.0)
u_in(t::Real) = x -> u_in(x, t)
u_noSlip(t::Real) = x -> u_noSlip(x, t)
∂tu_in(t) = x -> VectorValue(0.0, 0.0)
∂tu_in(x, t) = ∂tu_in(t)(x)
∂t(::typeof(u_in)) = ∂tu_in
∂t(::typeof(u_noSlip)) = ∂tu_in

## Domain
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
model = DiscreteModelFromFile("../models/cylinder_NSI.json")
labels = get_face_labeling(model)
writevtk(model, "model")

## Weak form
# Laws
@law conv(u, ∇u) = (∇u') * u
@law dconv(du, ∇du, u, ∇u) = conv(u, ∇du) + conv(du, ∇u)
@law σ_dev(ε) = 2 * ν * ε

# Bilinear function
function a(x, y)
    u, p = x
    v, q = y
    inner(ε(v), σ_dev(ε(u))) - (∇ * v) * p + q * (∇ * u)
end

# Nonlinear convective term
c(u, v) = v * conv(u, ∇(u))
dc(u, du, v) = v * dconv(du, ∇(du), u, ∇(u))

# Navier-Stokes residual
function res(t, x, xt, y)
    u, p = x
    ut, pt = xt
    v, q = y
    inner(ut, v) + a(x, y) + c(u, v)
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
    inner(dut, v)
end

## Solver functions
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

## PostProcessing
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

    # Surface triangulation
    trian_Γc = BoundaryTriangulation(model, "cylinder")
    quad_Γc = CellQuadrature(trian_Γc, 2)
    n_Γc = get_normal_vector(trian_Γc)

    # Drag & Lift coefficients
    coeff(F) = 2 * F / (ρ * Um^2 * ⌀)

    # Initialize arrays
    tpl = Real[]
    CDpl = Real[]
    CLpl = Real[]
    uhn = xh0[1]
    phn = xh0[2]

    # Get solution at n+θ (where pressure and velocity are balanced)
    θ = 0.5

    # Loop over steps
    for (xh, t) in sol

        # Get the solution at n+θ (where velocity and pressure are balanced)
        uh = xh.blocks[1]
        ph = xh.blocks[2]
        phθ = θ * ph + (1.0 - θ) * phn
        uh_Γc = restrict(uh, trian_Γc)
        uhn_Γc = restrict(uhn, trian_Γc)
        ph_Γc = restrict(phθ, trian_Γc)
        εθ = θ * ε(uh_Γc) + (1.0 - θ) * ε(uhn_Γc)
        FD, FL = sum(integrate(
            (σ_dev(ε(uh_Γc)) * n_Γc - ph_Γc * n_Γc),
            trian_Γc,
            quad_Γc,
        ))

        # Drag and lift coefficients
        push!(tpl, t)
        push!(CDpl, -coeff(FD))
        push!(CLpl, coeff(FL))

        # store step n
        uhn = uh
        phn = ph
    end

    return (tpl, CDpl, CLpl)
end


## Simutation function
function runCylinder(model::DiscreteModel, labels)

    ## FE spaces
    # Test FE spaces
    D = 2
    order = 2
    V = FESpace(
        reffe = :Lagrangian,
        conformity = :H1,
        valuetype = VectorValue{D,Float64},
        model = model,
        labels = labels,
        order = order,
        dirichlet_tags = ["inlet", "noslip", "cylinder"],
    )
    Q = TestFESpace(
        reffe = :Lagrangian,
        conformity = :H1,
        valuetype = Float64,
        model = model,
        order = order - 1,
        constraint = :zeromean,
    )

    # Trial FE spaces
    U = TransientTrialFESpace(V, [u_in, u_noSlip, u_noSlip])
    U0 = TrialFESpace(V, [u_in(t0), u_noSlip(t0), u_noSlip(t0)])
    P = TrialFESpace(Q)

    # Multifield FE spaces
    Y = MultiFieldFESpace([V, Q])
    X = MultiFieldFESpace([U, P])
    X0 = MultiFieldFESpace([U0, P])

    ## Triangulation and CellQuadrature
    trian = Triangulation(model)
    degree = (order - 1) * 2
    quad = CellQuadrature(trian, degree)

    ## FE operators
    # Navier-Stokes FE operator
    t_Ω = FETerm(res, jac, jac_t, trian, quad)
    op = TransientFEOperator(X, Y, t_Ω)
    # Stokes FE operator
    t_Stokes_Ω = LinearFETerm(a, trian, quad)
    op_Stokes = FEOperator(X0, Y, t_Stokes_Ω)

    # Stokes solution
    println("solveStokes")
    xh0 = solveStokes(op_Stokes)

    # Initialize Paraview files
    folderName = "ins-results"
    fileName = "fields"
    if !isdir(folderName)
        mkdir(folderName)
    end
    filePath = join([folderName, fileName], "/")

    # Output to Paraview
    #println("writeStokes")
    #writePVD(filePath, trian, [(xh0, 0.0)])

    # Transient Navier-Stokes solution (initial stage to reach fully developed flow)
    println("solveNavierStokes 1")
    sol_t = solveNavierStokes(op, xh0, 0.0, 8.0, 0.05, 0.5)

    # Output transient solution to Paraview
    #println("writeNavierStokes 1")
    #writePVD(filePath, trian, sol_t, append=true)

    # Get last solution snapshot as initial condition (this should be done overwriting Base.last)
    for (xh_tn, tn) in sol_t
        xh0 = xh_tn
    end

    # Transient Navier Stokes solution to pick statistics
    println("solveNavierStokes 2")
    sol_t = solveNavierStokes(op, xh0, 8.0, 10.0, 0.005, 0.5)

    # Output transient solution to Paraview
    #println("writeNavierStokes 2")
    #writePVD(filePath, trian, sol_t, append=true)

    # Output drag and lift coefficients
    (t, CD, CL) = computeForces(model, sol_t, xh0)
    return (t, CD, CL)

end

## Execute simulation
#(t, CD, CL) = @time runCylinder(model0, labels0)
(t, CD, CL) = @time runCylinder(model, labels)
p1 = plot(t, CD, label="CD")
p2 = plot(t, CL, label="CL")
display(plot(p1,p2,layout=(1,2)))
println("CDmax: ", maximum(CD), ", CLmax: ", maximum(CL))

## Results

# Velocity field animation
# ![](../assets/cylinder_ins/cylinder.gif)

# Drag and lift coefficients evolution

# |   | CD_max | CL_max |
# | computed | 3.178 | 1.012 |
# | Reference range | [3.220, 3.24] | [0.990,1.010] |
