# Driver that computes the lid-driven cavity benchmark at low Reynolds numbers
# when using a mixed FE Q(k)/Pdisc(k-1).

##
using Test
using Gridap
import Gridap: ∇

##
# Construct the discrete model for the domain Ω = [0,1]^2 and a structured mesh
# of 10 × 10 square elements
D = 2
n = 100
model = CartesianDiscreteModel(domain=(0.0,1.0,0.0,1.0), partition=(n,n))
# Construct the FEspace.
# The space is a vector Lagrangian space in 2D.
# We allow each one of the fields to have different boundary conditions
order = 2
const T = VectorValue{2,Float64}
# num_comps = 2
# 1,2,3,4 are the tags for the vertices of the square domain Ω, 5 is the bottom
# side, 6 the top side, 7 the left side, and 8 the right side.
diritags = [1,2,3,4,5,6,7,8]
# We want to enforce both u_x and u_y components for all entities in the
# Dirichlet boundary. The value to be enforced will be defined later.
# For the velocity, we use a C0 (continuous) FE space of Qk (tensor product of
# Pk uni-variate polynomials in each direction) with the previous Dirichlet
# boundary.
fespace1 = CLagrangianFESpace(T,model,order,diritags)
# For the pressure, we consider a Pk space at each cell, discontinuous among cells.
# Since the pressure space can only be defined up to a constant, we enforce
# an arbitrary dof to be zero.
# @santiagobadia : I would create a method to fix dof 1, nicer
# @santiagobadia : I would create a method for all FESpaces
reffe = PDiscRefFE(Float64,D,order-1)
_fespace2 = DiscFESpace(reffe,model)
fixeddofs = [1,]
fespace2 = ConstrainedFESpace(_fespace2,fixeddofs)

# Define test and trial
V1 = TestFESpace(fespace1)
V2 = TestFESpace(fespace2)
V = [V1, V2]

# Now, we define the value of the velocity to be enforced on the Dirichlet
# boundary
uD_1(x) = VectorValue(0.0,0.0)
uD_2(x) = VectorValue(1.0,0.0)
uD = [ (i == 6) ? uD_2 : uD_1 for i = 1:8 ]
U1 = TrialFESpace(fespace1,uD)
U2 = TrialFESpace(fespace2)
U = [U1, U2]

# Define integration mesh and quadrature for volume
trian = Triangulation(model)
quad = CellQuadrature(trian,order=(order-1)*2)

# divfun(x,∇u) = tr((∇u))
# div(u) = CellBasis(trian,divfun,∇(u))


# Terms in the volume
a(v,u) = inner(∇(v[1]),∇(u[1])) - inner(div(v[1]),u[2]) + inner(v[2],div(u[1]))
t_Ω = LinearFETerm(a,trian,quad)
op = LinearFEOperator(V,U,t_Ω)

# Now we compute the resulting FE problem
uh = solve(op)

# and write the results
writevtk(trian,"../tmp/stokesresults",cellfields=["uh"=>uh[1],"ph"=>uh[2]])
##
