module NavierStokesLidDrivenCavity

# Driver that computes the lid-driven cavity benchmark at low Reynolds numbers
# when using a mixed FE Q(k)/Pdisc(k-1).

##
using Test
using Gridap
import Gridap: ∇
using LineSearches: BackTracking

# The first part is identical to Stokes driver
D = 2
n = 100
model = CartesianDiscreteModel(domain=(0.0,1.0,0.0,1.0), partition=(n,n))
order = 2
const T = VectorValue{2,Float64}
diritags = [1,2,3,4,5,6,7,8]
fespace1 = CLagrangianFESpace(T,model,order,diritags)
reffe = PDiscRefFE(Float64,D,order-1)
_fespace2 = DiscFESpace(reffe,model)
fixeddofs = [1,]
fespace2 = ConstrainedFESpace(_fespace2,fixeddofs)
V1 = TestFESpace(fespace1)
V2 = TestFESpace(fespace2)
V = [V1, V2]
uD_1(x) = VectorValue(0.0,0.0)
uD_2(x) = VectorValue(1.0,0.0)
uD = [ (i == 6) ? uD_2 : uD_1 for i = 1:8 ]
U1 = TrialFESpace(fespace1,uD)
U2 = TrialFESpace(fespace2)
U = [U1, U2]
trian = Triangulation(model)
quad = CellQuadrature(trian,order=(order-1)*2)

# Reynolds number
const Re = 1.0
# santiagobadia: This is the way that I see I can implement the convection term with the
# current machinery
@law conv(x,u,∇u) = Re*adjoint(∇u)*u
@law conv2(x,∇u,u) = conv(x,u,∇u)
@law dconv(x,du,∇du,u,∇u) = conv(x,u,∇du)+conv(x,du,∇u)

# @santiagobadia : When the driver will work, I will do what we have said,
# @fverdugo: I guess that creating this closure at each integration point will be very expensive
# putting this in FieldValues module
# (*)(u::VectorValue,::typeof(gradient)) = (∇du) -> ugrad(u,∇du)
# function ugrad(u::VectorValue,∇du::VectorValue)
# VectorValue(   (u.array)' *  ∇du.array   )
# end

# Terms in the volume
a(v,u) = inner(∇(v[1]),∇(u[1])) - inner(div(v[1]),u[2]) + inner(v[2],div(u[1]))
c(v,u) = inner(v,conv(u,∇(u)))
#dc(v,du,u) = inner(v,dconv(du,∇(du),u,∇(u)))
dc(v,du,u) = inner(v, conv(du,∇(u))) + inner(v, conv2(∇(du),u))  #inner(v, conv(u,∇(du))) # + inner(v, conv(du,∇(u)))
res(u,v) = a(v,u) + c(v[1],u[1])
jac(u,v,du) = a(v,du) + dc(v[1],du[1],u[1])
t_Ω = NonLinearFETerm(res,jac,trian,quad)
assem = SparseMatrixAssembler(V,U)
op = NonLinearFEOperator(V,U,assem,t_Ω)

nls = JuliaNLSolver(
  show_trace=true,
  method=:newton,
  linesearch=BackTracking())

@show typeof(TrialFESpace(op))

solver = NonLinearFESolver(nls)
uh, ph = solve(solver,op)
# Now we compute the resulting FE problem

# and write the results
writevtk(trian,"ins-results",cellfields=["uh"=>uh,"ph"=>ph])
##
end # module
