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

limits = (0.0,1.0,0.0,1.0)
model = CartesianDiscreteModel(domain=limits, partition=(20,20))

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
op = LinearFEOperator(V0,U,t_Ω)

uh = solve(op)

e = u - uh

l2(u) = inner(u,u)
h1(u) = a(u,u) + l2(u)

el2 = sqrt(sum( integrate(l2(e),trian,quad) ))
eh1 = sqrt(sum( integrate(h1(e),trian,quad) ))

tol = 1.e-8
@assert el2 < tol
@assert eh1 < tol

const k = 2*pi

u(x) = sin(k*x[1]) * x[2]
∇u(x) = VectorValue(k*cos(k*x[1])*x[2], sin(k*x[1]))
f(x) = (k^2)*sin(k*x[1])*x[2]

∇(::typeof(u)) = ∇u

b(v) = inner(v,f)

function run(n,order)

  limits = (0.0,1.0,0.0,1.0)
  model = CartesianDiscreteModel(domain=limits, partition=(n,n))
  
  diritag = "boundary"
  V = CLagrangianFESpace(Float64,model,order,diritag)
  
  V0 = TestFESpace(V)
  U = TrialFESpace(V,u)
  
  trian = Triangulation(model)
  quad = CellQuadrature(trian,order=order+2)
  
  t_Ω = AffineFETerm(a,b,trian,quad)
  op = LinearFEOperator(V0,U,t_Ω)
  
  uh = solve(op)
  
  e = u - uh
  
  el2 = sqrt(sum( integrate(l2(e),trian,quad) ))
  eh1 = sqrt(sum( integrate(h1(e),trian,quad) ))

  (el2, eh1)

end

function conv_test(ns,order)

  el2s = Float64[]
  eh1s = Float64[]
  hs = Float64[]

  for n in ns

    el2, eh1 = run(n,order)
    h = 1.0/n

    push!(el2s,el2)
    push!(eh1s,eh1)
    push!(hs,h)

  end

  (el2s, eh1s, hs)

end

# Loooo

el2s, eh1s, hs = conv_test([8,16,32,64,128],2)

#src @show (log10(el2s[1]) - log10(el2s[end])) / (log10(hs[1]) - log10(hs[end]))
#src @show (log10(eh1s[1]) - log10(eh1s[end])) / (log10(hs[1]) - log10(hs[end]))

# Loooo

using Plots

plot(hs,[el2s eh1s],
    xaxis=:log, yaxis=:log,
    label=["L2" "H1"],
    shape=:auto,
    xlabel="h",ylabel="error norm")

#src savefig("conv.png")

#md # If you run the code in a notebook, you will see a figure like this one:
#md # ![](../assets/t002_validation/conv.png)

function slope(hs,errors)
  x = log10.(hs)
  y = log10.(errors)
  linreg = hcat(fill!(similar(x), 1), x) \ y
  linreg[2]
end

slope(hs,el2s)

slope(hs,eh1s)


