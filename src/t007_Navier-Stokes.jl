module NavierStokesLidDrivenCavity

# Driver that computes the lid-driven cavity benchmark at low Reynolds numbers
# when using a mixed FE Q(k)/Pdisc(k-1).

##
using Test
using Gridap
import Gridap: ∇
using LineSearches: BackTracking

D = 2
n = 100
model = CartesianDiscreteModel(domain=(0.0,1.0,0.0,1.0), partition=(n,n))
labels = FaceLabels(model)
add_tag_from_tags!(labels,"diri0",[1,2,3,4,5,7,8])
add_tag_from_tags!(labels,"diri1",[6,])

order = 2
const T = VectorValue{D,Float64}
diritags = ["diri0","diri1"]

fespace1 = FESpace(
  reffe=:Lagrangian,
  conformity=:H1,
  valuetype=T,
  model=model,
  labels=labels,
  order=order,
  diritags=diritags)

#reffe = PDiscRefFE(Float64,D,order-1)
#_fespace2 = DiscFESpace(reffe,model)
#fespace2 = ConstrainedFESpace(_fespace2,fixeddofs)

_fespace2 = FESpace(
  reffe=:PLagrangian,
  conformity=:L2,
  valuetype=Float64,
  model=model,
  order=order-1,
  constraint = :zeromean)

fixeddofs = [1,]
#fespace2 = ConstrainedFESpace(_fespace2,fixeddofs)
fespace2 = _fespace2

V = TestFESpace(fespace1)
Q = TestFESpace(fespace2)
Y = [V, Q]

uD0(x) = VectorValue(0.0,0.0)
uD1(x) = VectorValue(1.0,0.0)
U = TrialFESpace(fespace1,[uD0,uD1])
P = TrialFESpace(fespace2)
X = [U, P]

trian = Triangulation(model)
quad = CellQuadrature(trian,order=(order-1)*2)

const Re = 10.0
@law conv(x,u,∇u) = Re*(∇u')*u
@law dconv(x,du,∇du,u,∇u) = conv(x,u,∇du)+conv(x,du,∇u)

# Terms in the volume
function a(y,x)
  u, p = x
  v, q = y
  inner(∇(v),∇(u)) - inner(div(v),p) + inner(q,div(u))
end

c(v,u) = inner(v,conv(u,∇(u)))
dc(v,du,u) = inner(v,dconv(du,∇(du),u,∇(u)))

function res(x,y)
  u, p = x
  v, q = y
  a(y,x) + c(v,u)
end

function jac(x,y,dx)
  u, p = x
  v, q = y
  du, dp = dx
  a(y,dx)+ dc(v,du,u)
end

t_Ω = NonLinearFETerm(res,jac,trian,quad)
op = NonLinearFEOperator(Y,X,t_Ω)

#using Gridap.FEOperators: NonLinearOpFromFEOp
#algop = NonLinearOpFromFEOp(op)
#
#X = MultiFESpace(X)
#u = rand(num_free_dofs(X))
#du = rand(num_free_dofs(X))
#d = 0.000001
#
#e = residual(algop,u+d*du) - ( residual(algop,u) + d*jacobian(algop,u)*du )
#
#@show maximum(abs.(e))
#
#kk




nls = JuliaNLSolver(
  show_trace=true,
  method=:newton,
  linesearch=BackTracking())

solver = NonLinearFESolver(nls)
uh, ph = solve(solver,op)

# function At(δt,v,u,x0)
#   u0, p0 = x0
#   a(v,u) = inner(∇(v[1]),∇(u[1])) - inner(div(v[1]),u[2]) + inner(v[2],div(u[1]))
#   c(v,u) = inner(v,conv(u,∇(u)))
#   dc(v,du,u) = inner(v,dconv(du,∇(du),u,∇(u)))
#   res(v,u) = a(v,u) + c(v,u)
#   rest(v,u) = δt*res(v,u) + inner(v,u) - inner(v,u0)
#   jac(v,du,u) = a(v,du) + dc(v,du,u)
#   jact(v,du,u) = δt*jac(v,du,u) + inner(v,u0)
#   return rest, jact
# end
#
# Nt = 10
# Tf = 1.0
# δt = 1.0/Nt
# u0 = zero([U1,U2])
# for i in 1:10
#   rest, jact = At(δt,v,u,u0)
#   t_Ω = NonLinearFETerm(rest,jact,trian,quad)
#   op = NonLinearFEOperator(V,U,t_Ω)
#   uh, ph = solve(solver,op)
#   u0 = uh
# end



# Now we compute the resulting FE problem

# and write the results
writevtk(trian,"ins-results",cellfields=["uh"=>uh,"ph"=>ph])
##
end # module
