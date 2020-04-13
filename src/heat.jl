using Gridap
using GridapTimeStepper.ODETools
using GridapTimeStepper.TransientFETools
using Test

u(x,t) = (x[1] + x[2])*t
u(t::Real) = x -> u(x,t)
∇u(x,t) = VectorValue(1,1)*t
∇u(t::Real) = x -> ∇u(x,t)
import Gridap: ∇
∇(::typeof(u)) = ∇u
∇(u) === ∇u

∂tu(t) = x -> (x[1]+x[2])
import GridapTimeStepper.TransientFETools: ∂t
∂t(::typeof(u)) = ∂tu
@test ∂t(u) === ∂tu

f(t) = x -> (x[1]+x[2])

domain = (0,1,0,1)
partition = (2,2)
model = CartesianDiscreteModel(domain,partition)

order = 1
V0 = TestFESpace(
  reffe=:Lagrangian, order=order, valuetype=Float64,
  conformity=:H1, model=model, dirichlet_tags="boundary")

U = TransientTrialFESpace(V0,u)

trian = Triangulation(model)
degree = 2
quad = CellQuadrature(trian,degree)

a(u,v) = ∇(v)*∇(u)
b(v,t) = v*f(t)

res(t,u,ut,v) = a(u,v) + ut*v - b(v,t)
jac(t,u,ut,du,v) = a(du,v)
jac_t(t,u,ut,dut,v) = dut*v

t_Ω = FETerm(res,jac,jac_t,trian,quad)
op = TransientFEOperator(U,V0,t_Ω)

u0 = u(0.0)
t0 = 0.0
tF = 1.0
dt = 0.1

ls = LUSolver()
tol = 1.0
maxiters = 20
using Gridap.Algebra: NewtonRaphsonSolver
nls = NLSolver(ls;show_trace=true,method=:newton) #linesearch=BackTracking())

odes = BackwardEuler(nls,dt)
solver = TransientFESolver(odes)

uh0 = interpolate_everywhere(U(0.0),u(0.0))
sol_t = solve(solver,op,uh0,t0,tF)
sol_t = solve(solver,op,uh0,t0,tF)

_t_n = t0
for (uh_tn, tn) in sol_t
  global _t_n
  _t_n += dt
  e = u(tn) - uh_tn
  el2 = sqrt(sum( integrate(l2(e),trian,quad) ))
  @test el2 < tol
#   # writevtk(trian,"sol at time: $tn",cellfields=["u" => uh_tn])
end
