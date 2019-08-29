# # Tutorial 2: Code validation
#
#md # [![](https://mybinder.org/badge_logo.svg)](@__BINDER_ROOT_URL__/notebooks/t002_validation.ipynb)
#md # [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/notebooks/t002_validation.ipynb)
# 
# ## Learning outcomes
#
# - How to use the method of manufactured solutions
# - How to perform a convergence test
# - How to define the discretization error
# - How to integrate error norms
# - How to generate Cartesian meshes in arbitrary dimensions
#
# ## Problem statement

# Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.

using Gridap

u(x) = x[1] + x[2]
∇u(x) = VectorValue(1.0,1.0)
f(x) = 0.0

import Gridap: ∇
∇(::typeof(u)) = ∇u

model = CartesianDiscreteModel(domain=(0.0,1.0,0.0,1.0), partition=(4,4))

order = 1
diritag = "boundary"
V = CLagrangianFESpace(Float64,model,order,diritag)

V0 = TestFESpace(V)
U = TrialFESpace(V,u)

trian = Triangulation(model)
quad = CellQuadrature(trian,order=2)

a(v,u) = inner(∇(v), ∇(u))
b(v) = inner(v,f)

t_Ω = AffineFETerm(a,b,trian,quad)
assem = SparseMatrixAssembler(V0,U)
op = LinearFEOperator(V0,U,assem,t_Ω)

#TODO
#op = LinearFEOperator(V0,U,t_Ω)

ls = LUSolver()
solver = LinearFESolver(ls)
uh = solve(solver,op)

#TODO
#uh = solve(op)

e = CellField(trian,u) - uh

#TODO
#e = u - uh

l2(u) = inner(u,u)
h1(u) = a(u,u) + l2(u)

el2 = sqrt(sum( integrate(l2(e),trian,quad) ))
eh1 = sqrt(sum( integrate(h1(e),trian,quad) ))

tol = 1.e-8
@assert el2 < tol
@assert eh1 < tol

