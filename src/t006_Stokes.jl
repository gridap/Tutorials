# Driver that computes the lid-driven cavity benchmark at low Reynolds numbers
# when using a mixed FE Q(k)/Pdisc(k-1).

# Load Gridap library
using Gridap

# Discrete model
n = 100
domain = (0,1,0,1)
partition = (n,n)
model = CartesianDiscreteModel(domain, partition)

# Define Dirichlet boundaries
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"diri1",[6,])
add_tag_from_tags!(labels,"diri0",[1,2,3,4,5,7,8])

# Define test FESpaces (Q2/P1(disc) pair)
V = TestFESpace(
  reffe=:QLagrangian, conformity=:H1, valuetype=VectorValue{2,Float64},
  model=model, labels=labels, order=2, dirichlet_tags=["diri0","diri1"])
Q = TestFESpace(
  reffe=:PLagrangian, conformity=:L2, valuetype=Float64,
  model=model, order=1, constraint=:zeromean)
Y = MultiFieldFESpace([V,Q])

# Define trial FESpaces from Dirichlet values
u0 = VectorValue(0,0)
u1 = VectorValue(1,0)
U = TrialFESpace(V,[u0,u1])
P = TrialFESpace(Q)
X = MultiFieldFESpace([U,P])

# Define integration mesh and quadrature
trian = get_triangulation(model); degree = 2
quad = CellQuadrature(trian,degree)

# Define and solve the FE problem
function a(y,x)
  v,q = y
  u,p = x
  inner(∇(v),∇(u)) - (∇*v)*p + q*(∇*u)
end
t_Ω = LinearFETerm(a,trian,quad)
op = AffineFEOperator(Y,X,t_Ω)
uh, ph = solve(op)

# Export results to vtk
writevtk(trian,"results",order=2,cellfields=["uh"=>uh,"ph"=>ph])
