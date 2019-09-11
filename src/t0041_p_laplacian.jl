# # Tutorial 4: p-Laplacian
#
#md # [![](https://mybinder.org/badge_logo.svg)](@__BINDER_ROOT_URL__/notebooks/t0041_p_laplacian.ipynb)
#md # [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/notebooks/t0041_p_laplacian.ipynb)
# 
# ## Problem statement

using Gridap
using LineSearches: BackTracking
using LinearAlgebra: norm
import Random
Random.seed!(1234)

const p = 3

@law j(x,∇u) = norm(∇u)^(p-2) * ∇u

@law dj(x,∇du,∇u) = (p-2)*norm(∇u)^(p-4)*inner(∇u,∇du)*∇u + norm(∇u)^(p-2) * ∇du

f(x) = 1.0

res(u,v) = inner( ∇(v), j(∇(u)) ) - inner(v,f)

jac(u,v,du) = inner(  ∇(v) , dj(∇(du),∇(u)) )

model = DiscreteModelFromFile("../models/model.json");

labels = FaceLabels(model)

add_tag_from_tags!(labels,"diri0",["sides", "sides_c"])
add_tag_from_tags!(labels,"diri1",
  ["circle","circle_c", "triangle", "triangle_c", "square", "square_c"])

order = 1
diritags = ["diri0", "diri1"]
V = CLagrangianFESpace(Float64,model,labels,order,diritags);

V0 = TestFESpace(V)
U = TrialFESpace(V,[0.0,1.0]);

trian = Triangulation(model)
quad = CellQuadrature(trian,order=2)

t_Ω = NonLinearFETerm(res,jac,trian,quad)
op = NonLinearFEOperator(V,U,t_Ω)

# Setup linear solver
ls = BackslashSolver()

# Setup non-linear solver
nls = JuliaNLSolver(
  ls;
  show_trace=true,
  store_trace=true,
  method=:newton,
  linesearch=BackTracking())

solver = NonLinearFESolver(nls)

x = rand(Float64,num_free_dofs(U))

uh = FEFunction(U,x)

cache = solve!(uh,solver,op)

writevtk(trian,"results",cellfields=["uh"=>uh])

# ![](../assets/t0041_p_laplacian/sol-plap.png)
