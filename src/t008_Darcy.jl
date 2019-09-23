module DarcyDriver

# Testing Darvy problem with Dirichlet BCs for the velocity
# and pressure prescribed in a single point
##
using Test
using Gridap
import Gridap: ∇

# Construct the discrete model
model = CartesianDiscreteModel(domain=(0.0,1.0,0.0,1.0), partition=(50,50))

# Construct the mixed FE spaces
order = 2
V = FESpace( reffe=:RaviartThomas, conformity=:HDiv, valuetype = Float64, order=2, model=model, diritags = [5,6])
_Q = FESpace( reffe=:PLagrangian, conformity=:L2, valuetype = Float64, order = 1, model = model)

# Define test and trial
V_0 = TestFESpace(V)
Q = TestFESpace(_Q) # Since no bc's the trial space is the same
Y = [V_0, Q]

const ν = 1000.0

ue(x) = VectorValue(ν, 0.0)
pe(x) = 1.0-x[1]

V_g = TrialFESpace(V,ue)
# Q_trial = TrialFESpace(Q)
X = [V_g, Q]


# Define integration mesh and quadrature for volume
trian = Triangulation(model)
quad = CellQuadrature(trian,order=2)

const kinv_1 = TensorValue(1.0,0.0,0.0,1.0)
const kinv_2 = TensorValue(1.0/ν,0.0,0.0,1.0)
# const kinv_2 = TensorValue(100.0,90.0,90.0,100.0)

@law function σ(x,u)
 if ((abs(x[1]-0.5) <= 0.1) && (abs(x[2]-0.5) <= 0.1))
   return kinv_1*u
 else
   return kinv_2*u
 end
end
#
# @law σ(x,u) = kinv_1*u

a(v,u) =
   inner(v[1],σ(u[1])) - inner(div(v[1]),u[2]) + inner(v[2],div(u[1]))
t_Ω = LinearFETerm(a,trian,quad)

# Setup integration on Neumann boundary
neumanntags = [7,]
btrian = BoundaryTriangulation(model,neumanntags)
bquad = CellQuadrature(btrian,order=order*2)
nb = NormalVector(btrian)

gfun(x) = -1.0
g(v) = inner(v[1]*nb,gfun)
# Define weak form terms
t_ΓN = FESource(g,btrian,bquad)

# Define Assembler
assem = SparseMatrixAssembler(Y,X)

# Define FE problem
op = LinearFEOperator(Y,X,assem,t_Ω,t_ΓN)

# Solve!
xh = solve(op)

# Define exact solution and error
# eu = ue - xh[1]

# ep = pe - xh[2]

# hdiv(u) = inner(div(u),div(u)) + l2(u)

# e1l2 = sqrt(sum( integrate(l2(u1),trian,quad) ))
# Compute errors
# e1l2 = sqrt(sum( integrate(l2(eu),trian,quad) ))
# e1h1 = sqrt(sum( integrate(hdiv(e1),trian,quad) ))
# e2l2 = sqrt(sum( integrate(l2(ep),trian,quad) ))

# xh[1].free_dofs

# @test e1l2 < 1.e-8
# @test e1hdiv < 1.e-8

# @test e2l2 < 1.e-8

writevtk(trian,"darcyresults",cellfields=["uh"=>xh[1],"ph"=>xh[2]])
##
end # module
