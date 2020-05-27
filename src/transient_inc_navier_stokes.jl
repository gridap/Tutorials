# In this tutorial, we will learn
#  - How to solve nonlinear multi-field PDEs in Gridap
#  - How to build FE spaces whose functions have zero mean value
#
# ## Problem statement
#
# The goal of this last tutorial is to solve a nonlinear multi-field PDE. As a model problem, we consider a well known benchmark in computational fluid dynamics, the lid-driven cavity for the incompressible Navier-Stokes equations. Formally, the PDE we want to solve is: find the velocity vector $u$ and the pressure $p$ such that
#
# ```math
# \left\lbrace
# \begin{aligned}
# -\Delta u + \mathit{Re}\ (u\cdot \nabla)\ u + \nabla p = 0 &\text{ in }\Omega,\\
# \nabla\cdot u = 0 &\text{ in } \Omega,\\
# u = g &\text{ on } \partial\Omega,
# \end{aligned}
# \right.
# ```
#
# where the computational domain is the unit square $\Omega \doteq (0,1)^d$, $d=2$, $\mathit{Re}$ is the Reynolds number (here, we take $\mathit{Re}=10$), and $(w \cdot \nabla)\ u = (\nabla u)^t w$  is the well known convection operator. In this example, the driving force is the Dirichlet boundary velocity $g$, which is a non-zero horizontal velocity with a value of $g = (1,0)^t$ on the top side of the cavity, namely the boundary $(0,1)\times\{1\}$, and $g=0$ elsewhere on $\partial\Omega$. Since we impose Dirichlet boundary conditions on the entire boundary $\partial\Omega$, the mean value of the pressure is constrained to zero in order have a well posed problem,
#
# ```math
# \int_\Omega q \ {\rm d}\Omega = 0.
# ```
#
# ## Numerical Scheme
#
# In order to approximate this problem we chose a formulation based on inf-sub stable $Q_k/P_{k-1}$ elements with continuous velocities and discontinuous pressures (see, e.g., [1] for specific details). The interpolation spaces are defined as follows.  The velocity interpolation space is
#
# ```math
# V \doteq \{ v \in [C^0(\Omega)]^d:\ v|_T\in [Q_k(T)]^d \text{ for all } T\in\mathcal{T} \},
# ```
# where $T$ denotes an arbitrary cell of the FE mesh $\mathcal{T}$, and $Q_k(T)$ is the local polynomial space in cell $T$ defined as the multi-variate polynomials in $T$ of order less or equal to $k$ in each spatial coordinate. Note that, this is the usual continuous vector-valued Lagrangian FE space of order $k$ defined on a mesh of quadrilaterals or hexahedra.  On the other hand, the space for the pressure is
#
# ```math
# \begin{aligned}
# Q_0 &\doteq \{ q \in Q: \  \int_\Omega q \ {\rm d}\Omega = 0\}, \text{ with}\\
# Q &\doteq \{ q \in L^2(\Omega):\ q|_T\in P_{k-1}(T) \text{ for all } T\in\mathcal{T}\},
# \end{aligned}
# ```
# where $P_{k-1}(T)$ is the polynomial space of multi-variate polynomials in $T$ of degree less or equal to $k-1$. Note that functions in $Q_0$ are strongly constrained to have zero mean value. This is achieved in the code by removing one degree of freedom from the (unconstrained) interpolation space $Q$ and  adding a constant to the computed pressure so that the resulting function has zero mean value.
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
# In order to solve this nonlinear weak equation with a Newton-Raphson method, one needs to compute the Jacobian associated with the residual $r$. In this case, the Jacobian $j$ evaluated at a pair $(u,p)$ is the bilinear form defined as
#
# ```math
# [j(u,p)]((\delta u, \delta p),(v,q)) \doteq a((\delta u,\delta p),(v,q))  + [{\rm d}c(u)](\delta u,v),
# ```
# where ${\rm d}c$ results from the linearization of the convective term, namely
# ```math
# [{\rm d}c(u)](\delta u,v) \doteq \int_{\Omega} v \cdot \left( (u\cdot\nabla)\ \delta u \right) \ {\rm d}\Omega + \int_{\Omega} v \cdot \left( (\delta u\cdot\nabla)\ u \right)  \ {\rm d}\Omega.
# ```
# The implementation of this numerical scheme is done in Gridap by combining the concepts previously seen for single-field nonlinear PDEs  and linear multi-field problems.
#
# ## Discrete model
#
# We start with the discretization of the computational domain. We consider a $100\times100$ Cartesian mesh of the unit square.

using Gridap
using ForwardDiff
using LinearAlgebra
using Test
using GridapODEs.ODETools
using GridapODEs.TransientFETools
using Gridap.FESpaces: get_algebraic_operator
using Printf

import GridapODEs.TransientFETools: ∂t

# Problem parameters
const Um = 1.0
const H = 0.41
const ⌀ = 0.1
const ρ = 1.0

# Boundary conditions
u_in(x,t) = VectorValue( 1.5 * Um * x[2] * ( H - x[2] ) / ( (H/2)^2 ), 0.0 )
u_noSlip(x,t) = VectorValue( 0.0, 0.0 )
u_in(t::Real) = x -> u_in(x,t)
u_noSlip(t::Real) = x -> u_noSlip(x,t)
∂tu_in(t) = x -> VectorValue(0.0,0.0)
∂tu_in(x,t) = ∂tu_in(t)(x)
∂t(::typeof(u_in)) = ∂tu_in
∂t(::typeof(u_noSlip)) = ∂tu_in

# Domain
#n = 4
#domain = (0,1,0,1)
#partition = (n,n)
#model = CartesianDiscreteModel(domain,partition)
#labels = get_face_labeling(model)
#add_tag_from_tags!(labels,"diri1",[6,])
#add_tag_from_tags!(labels,"diri0",[1,2,3,4,5,7,8])

model = DiscreteModelFromFile("../models/cylinder_NSI.json")
writevtk(model,"model")

# Time stepping settings
t0 = 0.0
tF = 0.1
dt = 0.05
θ = 0.5

# ## FE spaces
D = 2
order = 2
V = FESpace(
  reffe=:Lagrangian, conformity=:H1, valuetype=VectorValue{D,Float64},
  model=model, order=order, dirichlet_tags=["inlet","noslip","cylinder"])
Q = TestFESpace(
  reffe=:Lagrangian, conformity=:H1, valuetype=Float64,
  model=model, order=order-1, constraint=:zeromean)



p(x,t) = 0.0
p(t::Real) = x -> p(x,t)

function u_t0(x,t)
    if(x[2]==1.0 && x[1]>0.0 && x[1]<1.0)
        return VectorValue(1.0,0.0)
    else
        return VectorValue(0.0,0.0)
    end
end
u_t0(t::Real) = x->u_t0(x,t)

U = TransientTrialFESpace(V,[u_in,u_noSlip,u_noSlip])
U0 = TrialFESpace(V,[u_in(t0),u_noSlip(t0),u_noSlip(t0)])
P = TrialFESpace(Q)

Y = MultiFieldFESpace([V, Q])
X = MultiFieldFESpace([U, P])
X0 = MultiFieldFESpace([U0, P])

# Initial FE spaces
#U0 = U(t0)
#P0 = P(t0)
#X0 = X(t0)

# ## Navier-Stokes weak form
@law conv(u,∇u) = (∇u')*u
@law dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)

const Re = 100.0
const ν = Um*⌀ / Re
@law σ_dev(ε) = 2*ν*ε
function a(x,y)
  u, p = x
  v, q = y
  inner(ε(v),σ_dev(ε(u))) - (∇*v)*p + q*(∇*u)
end

c(u,v) = v*conv(u,∇(u))
dc(u,du,v) = v*dconv(du,∇(du),u,∇(u))

function res(t,x,xt,y)
  u, p = x
  ut, pt = xt
  v, q = y
  inner(ut,v) + a(x,y) + c(u,v)
end

function jac(t,x,xt,dx,y)
  u, p = x
  v, q = y
  du, dp = dx
  a(dx,y)+ dc(u,du,v)
end

function jac_t(t,x,xt,dxt,y)
    dut, dpt = dxt
    v, q = y
    inner(dut,v)
end

# Triangulation and CellQuadrature
trian = Triangulation(model)
degree = (order-1)*2
quad = CellQuadrature(trian,degree)

# Navier-Stokes FE operator
t_Ω = FETerm(res,jac,jac_t,trian,quad)
op = TransientFEOperator(X,Y,t_Ω)

# Stokes FE operator
t_Stokes_Ω = LinearFETerm(a,trian,quad)
op_Stokes = FEOperator(X0,Y,t_Stokes_Ω)

# Stokes solution
ls = LUSolver()
solver_Stokes = FESolver(ls)
xh0 = solve(solver_Stokes,op_Stokes)
uh0 = xh0[1]
ph0 = xh0[2]

l2(w) = w*w
#println("L2-norm: ", sqrt(sum( integrate(l2(uh0),trian,quad) )))

# Transient Navier-Stokes solution
using LineSearches: BackTracking
nls = NLSolver(show_trace=true, method=:newton, linesearch=BackTracking())
odes = ThetaMethod(nls,dt,θ)
solver = TransientFESolver(odes)
sol_t = solve(solver,op,xh0,t0,tF)

# Post-processing

# Cylinder triangulation
trian_Γc = BoundaryTriangulation(model,"cylinder")
quad_Γc = CellQuadrature(trian_Γc,degree)
n_Γc = get_normal_vector(trian_Γc)

# Drag & Lift coefficients
coeff(F) = 2*F / (ρ*Um^2*⌀)
using Plots
tpl = [0.0]
CLpl = [0.0]
pl = plot(tpl,CLpl,leg=false)
anim = Animation()

_t_n = t0
_phn = ph0
_uhn = uh0

# Paraview files
outputName = "ins-results"
using WriteVTK
pvd = paraview_collection(outputName)
pvd[0.0] = writevtk(trian,outputName,cellfields=["uh_tn"=>uh0,"ph_tn"=>ph0])[1]

# import Base.push!
# function push!(output::vtkOutput, vtkFile::Array{String,1}, time::Float64)
#     append!(output.vtkFiles, vtkFile)
#     push!(output.times, time)
# end
# function writeFiles(output::vtkOutput)
#     if !isdir(output.outputName)
#         mkdir(output.outputName)
#     end
#     fileName = output.outputName * "/" * output.outputName * ".pvd"
#
# end

#push!(vtkFiles,writevtk(trian,"ins-result0.0",cellfields=["uh_tn"=>uh0,"ph_tn"=>ph0]))
#mkdir(outputName)
#vtkFiles = writevtk(trian,"ins-result0.0",cellfields=["uh_tn"=>uh0,"ph_tn"=>ph0]))

for (xh_tn, tn) in sol_t
  global _t_n
  global _phn
  global _uhn
  _t_n += dt
  uh_tn = xh_tn.blocks[1]
  ph_tn = xh_tn.blocks[2]
  uhθ = θ*uh_tn + (1.0 - θ)*_uhn
  phθ = θ*ph_tn + (1.0 - θ)*_phn
  _uhn = uh_tn
  _phn = ph_tn
  pvd[_t_n] = writevtk(trian,outputName,cellfields=["uh_tn"=>uh_tn,"ph_tn"=>ph_tn])
  #println("L2-norm: ", sqrt(sum( integrate(l2(uh_tn),trian,quad) )))
  uh_Γc = restrict(uh_tn,trian_Γc)
  ph_Γc = restrict(phθ,trian_Γc)
  FD, FL = sum( integrate( (σ_dev(ε(uh_Γc))*n_Γc - ph_Γc*n_Γc), trian_Γc, quad_Γc ) )
  println("CD: ", coeff(FD))
  println("CL: ", coeff(FL))
  push!(tpl,_t_n)
  push!(CLpl,coeff(FL))
  push!(pl, tpl, CLpl)
  frame(anim)
end
anim = @animate for i in 2:length(tpl)
    plot(tpl[1:i],CLpl[1:i])
end
vtk_save(pvd)
gif(anim,fps=5)
#plot(tpl,CLpl)
