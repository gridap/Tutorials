module DarcyDriver

# Testing Darvy problem with Dirichlet BCs for the velocity
# and pressure prescribed in a single point
##
using Test
using Gridap
import Gridap: ∇

u(x) = VectorValue(0.0,0.0)
∇u(x) = TensorValue(0.0,0.0,0.0,0.0)
∇(::typeof(u)) = ∇u

p(x) = x[1]
∇p(x) = VectorValue(1.0,0.0)
∇(::typeof(p)) = ∇p

model = CartesianDiscreteModel(domain=(0.0,1.0,0.0,1.0), partition=(50,50))

order = 2
V = FESpace( reffe=:RaviartThomas, conformity=:HDiv, order=2, model=model, diritags = [5,6])
_Q = FESpace( reffe=:QLagrangian, conformity=:L2, valuetype = Float64, order = 1,
                    model = model)

V_0 = TestFESpace(V)
Q = TestFESpace(_Q)
Y = [V_0, Q]

V_g = TrialFESpace(V,u)
X = [V_g, Q]

trian = Triangulation(model)
quad = CellQuadrature(trian,degree=2)

const ν = 100.0
const kinv_1 = TensorValue(1.0,0.0,0.0,1.0)
const kinv_2 = TensorValue(1.0/ν,0.0,0.0,1.0)

@law function σ(x,u)
 if ((abs(x[1]-0.5) <= 0.1) && (abs(x[2]-0.5) <= 0.1))
   return kinv_1*u
 else
   return kinv_2*u
 end
end

function a(y,x)
  v, q = y
  u, p = x
  inner(v,σ(u)) - inner(div(v),p) + inner(q,div(u))
end

t_Ω = LinearFETerm(a,trian,quad)

neumanntags = [7,8]
btrian = BoundaryTriangulation(model,neumanntags)
bquad = CellQuadrature(btrian,degree=order*2)
nb = NormalVector(btrian)

function b_Γ(y)
  v, q = y
  -inner(v*nb,p)
end

t_Γ = FESource(b_Γ,btrian,bquad)

op = LinearFEOperator(Y,X,t_Ω,t_Γ)

uh, ph = solve(op)

e1 = u - uh
e2 = p - ph

l2(u) = inner(u,u)
hdiv(u) = inner(div(u),div(u)) + l2(u)

# writevtk(trian,"/home/santiago/github-repos/Gridap/tmp/darcyresults",
         # cellfields=["uh"=>uh,"ph"=>ph])

##
end # module
