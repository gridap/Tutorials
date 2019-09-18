module DarcyDriver

# Testing Darvy problem with Dirichlet BCs for the velocity
# and pressure prescribed in a single point
##
using Test
using Gridap
import Gridap: ∇

const T = VectorValue{2,Float64}

# Construct the discrete model
model = CartesianDiscreteModel(domain=(0.0,1.0,0.0,1.0), partition=(2,2))

# Construct the FEspace 1
order = 2
D = 2
p = Polytope(fill(HEX_AXIS,D)...)
reffe = RaviartThomasRefFE(p,order)

grid = Grid(model,D)
trian = Triangulation(grid)
graph = GridGraph(model)
labels = FaceLabels(model)
diritags = [5,6]

# @santiagobadia : Again, find a general constructor for all FE spaces
fespace1 = ConformingFESpace(reffe,trian,graph,labels,diritags)

# Construct the FEspace 2
reffe = PDiscRefFE(Float64,D,order-1)
fespace2 = DiscFESpace(reffe,model)

# Define test and trial
V1 = TestFESpace(fespace1)
V2 = TestFESpace(fespace2)
V = [V1, V2]



ν = 1.0
# @santiagobadia : It produces an error that should be fixed
# u1(x) = ν*VectorValue(1.0,0.0)
# By the way if we put (10,0.0) it does not work either
u1(x) = VectorValue(1.0, 0.0)
u2(x) = 1.0-x[1]
U1 = TrialFESpace(fespace1,u1)
U2 = TrialFESpace(fespace2)
U = [U1, U2]

# Define integration mesh and quadrature for volume
trian = Triangulation(model)
quad = CellQuadrature(trian,order=2)

# @santiagobadia : If I eliminate u in the interface it does not work.
# It has full sense to define a law that only depens on the space corordinates

kinv_1 = TensorValue(1.0,0.0,0.0,1.0)
kinv_2 = TensorValue(100.0,90.0,90.0,100.0)

# @law function σ(x,u)
#   if ((abs(x[1]-0.5) <= 0.25) && (abs(x[2]-0.5) <= 0.25))
#     return kinv_1*u
#   else
#     return kinv_2*u
#   end
# end
# @law σ(x,u) = kinv_1*u

# @santiagobadia : If I use this law I get an error
@law σ(x,u) = kinv_1*u
# a(v,u) =
#   inner(v[1],σ(u[1])) - inner(div(v[1]),u[2]) + inner(v[2],div(u[1]))

# So, I am using now
a(v,u) =
   inner(v[1],kinv_1*(u[1])) - inner(div(v[1]),u[2]) + inner(v[2],div(u[1]))
# b(v) = inner(v[1],b1) + inner(v[2],b2)
# t_Ω = AffineFETerm(a,b,trian,quad)
t_Ω = LinearFETerm(a,trian,quad)


# Setup integration on Neumann boundary
neumanntags = [7,]
btrian = BoundaryTriangulation(model,neumanntags)
bquad = CellQuadrature(btrian,order=order*2)
nb = NormalVector(btrian)

# Integrand of the Neumann BC
# gfun(x) = VectorValue(0.0,0.0)
# gfun(x) = VectorValue(0.0,0.0)
# g(v) = inner(v[1],gfun)

# @santiagobadia : Doing this I get an error
gfun(x) = -1.0
g(v) = inner(v[1]*nb,gfun)


# Define weak form terms
t_ΓN = FESource(g,btrian,bquad)

# Define Assembler
assem = SparseMatrixAssembler(V,U)

# Define FE problem
op = LinearFEOperator(V,U,assem,t_Ω,t_ΓN)

# Solve!
uh = solve(op)

# Define exact solution and error
e1 = u1 - uh[1]

e2 = u2 - uh[2]

uh[1].free_dofs

# Define norms to measure the error
l2(u) = inner(u,u)
# hdiv(u) = inner(div(u),div(u)) + l2(u)

# Compute errors
e1l2 = sqrt(sum( integrate(l2(e1),trian,quad) ))
# e1h1 = sqrt(sum( integrate(hdiv(e1),trian,quad) ))
uh[1].free_dofs
e2l2 = sqrt(sum( integrate(l2(e2),trian,quad) ))

@test e1l2 < 1.e-8
# @test e1hdiv < 1.e-8

@test e2l2 < 1.e-8

writevtk(trian,"../tmp/darcyresults",cellfields=["uh"=>uh[1],"ph"=>uh[2]])
##
end # module
