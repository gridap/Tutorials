#md # !!! note
#
#     This tutorial is under construction, but the code below is already functional.
#
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

# Define reference FE (Q2/P1(disc) pair)
order = 2
reffeᵤ = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
reffeₚ = ReferenceFE(lagrangian,Float64,order-1;space=:P)

# Define test FESpaces
V = TestFESpace(model,reffeᵤ,labels=labels,dirichlet_tags=["diri0","diri1"],conformity=:H1)
Q = TestFESpace(model,reffeₚ,conformity=:L2,constraint=:zeromean)
Y = MultiFieldFESpace([V,Q])

# Define trial FESpaces from Dirichlet values
u0 = VectorValue(0,0)
u1 = VectorValue(1,0)
U = TrialFESpace(V,[u0,u1])
P = TrialFESpace(Q)
X = MultiFieldFESpace([U,P])

# Define triangulation and integration measure
degree = order
Ωₕ = Triangulation(model)
dΩ = Measure(Ωₕ,degree)

# Define bilinear and linear form
f = VectorValue(0.0,0.0)
a((u,p),(v,q)) = ∫( ∇(v)⊙∇(u) - (∇⋅v)*p + q*(∇⋅u) )dΩ
l((v,q)) = ∫( v⋅f )dΩ

# Build affine FE operator
op = AffineFEOperator(a,l,X,Y)

# Solve
uh, ph = solve(op)

# Export results to vtk
writevtk(Ωₕ,"results",order=2,cellfields=["uh"=>uh,"ph"=>ph])
