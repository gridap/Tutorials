
using Gridap

u(x) = x[1] + x[2]
f(x) = 0.0

limits = (0.0, 1.0, 0.0, 1.0)
model = CartesianDiscreteModel(domain=limits, partition=(4,4))

order = 1
V = DLagrangianFESpace(Float64,model,order)

V0 = TestFESpace(V)
U = TrialFESpace(V)

trian = Triangulation(model)
quad = CellQuadrature(trian,order=2)

btrian = BoundaryTriangulation(model,"boundary")
bquad = CellQuadrature(btrian,order=2)

strian = BoundaryTriangulation(model,"interior")
squad = CellQuadrature(strian,order=2)

a_Ω(v,u) = inner(∇(v), ∇(u))
b_Ω(v) = inner(v,f)
t_Ω = AffineFETerm(a_Ω,b_Ω,trian,quad)

a_∂Ω(v,u) = inner(v, ∇(u)*n ) - inner(∇(v)*n, u) + (γ/h) * inner(v,u)
b_∂Ω(v) = (γ/h) * inner(v,g) - inner(∇(v)*n, g)
t_∂Ω = AffineFETerm(a_∂Ω,b_∂Ω,trian,quad)

a_Γ(v,u) = 
  inner( jump(v*n), mean(∇(u)) ) -
  inner( mean(∇(v)), jump(u*n) ) + 
  (γ/h) * inner( jump(v*n), jump(u*n))

t_Γ = LinearFEOperator(a_Γ,strian,squad)



