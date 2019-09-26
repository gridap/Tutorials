# # Tutorial 6: Poisson equation (with DG)
#
#md # [![](https://mybinder.org/badge_logo.svg)](@__BINDER_ROOT_URL__/notebooks/t005_dg_discretization.ipynb)
#md # [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/notebooks/t005_dg_discretization.ipynb)
#
# ## Learning outcomes
#
# - How to solve a simple with a Discontinuous Galerkin (DG) discretization
# - How to build discontinuous FE spaces
# - How to integrate quantities on the mesh skeleton
# - How to compute jumps and averages of quantities on the mesh skeleton
#
#
# ## Problem statement
#
# The goal of this tutorial is to solve a simple PDE using a Discontinuous Galerkin (DG) formulation. For simplicity, we take the Poisson equation on the unit multi dimensional cube $\Omega \doteq (0,1)^d$, with $d=2$ and $d=3$, as a model problem: 
#
#
# ```math
# \left\lbrace
# \begin{aligned}
# -\Delta u = f  \ \text{in} \ \Omega\\
# u = g \ \text{on}\ \partial\Omega,\\
# \end{aligned}
# \right.
# ```
# where $f$ is the source term and $g$ is the Dirichlet boundary value.
#
#  We are going to solve two version of this problem. In a first stage, we take $d=3$ and consider a manufactured solution, namely $u(x) = 3 x_1 + x_2 + 2 x_3$, that belongs to the FE interpolation that we will build below. In this case, we expect to compute a numerical solution with an approximation error close to the machine precision. On the other hand, we will perform a convergence test for the 2D case ($d=2$). To this, end we will consider a manufactured solution that cannot by represented exactly by the interpolation, namely $u(x)=x_2 \sin(2 \pi\ x_1)$. Our goal is to confirm that the convergence order of the discretization error is the optimal one.
#
# ## Numerical Scheme
#
# In contrast to previous tutorials, we consider a DG formulation to approximate the problem. For the sake of simplicity, we take the well know (symmetric) interior penalty method. For this formulation, the approximation space is made of discontinuous piece-wise polynomials, namely
#
# ```math
# V \doteq \{ v\in L^2(\Omega):\ v|_{T}\in Q_p(T) \text{ for all } T\in\mathcal{T}  \},
# ```
# where $\mathcal{T}$ is the set of all cells $T$ of the FE mesh, and $Q_p(T)$ is a polynomial space of degree $p$ defined on a generic cell $T$. For simplicity, we consider Cartesian meshes in this tutorial. In this case, the space $Q_p(T)$ is made of multi-variate polynomials up to degree $p$ in each spatial coordinate. 
#
#
#
#
# In order to write the weak form of the problem, we need to introduce the set of interior and boundary facets associated with the FE mesh, denoted here as $\mathcal{F}_\Gamma$ and $\mathcal{F}_{\partial\Omega}$ respectively. In addition, for a given function $v\in V$ restricted to the interior facets $\mathcal{F}_\Gamma$, we need to define the well known jump and mean value operators:
# ```math
# \lbrack\!\lbrack v\ n \rbrack\!\rbrack \doteq v^+\ n^+ + v^- n^-, \text{ and } \{\! \!\{\nabla v\}\! \!\} \doteq \dfrac{ \nabla v^+ + \nabla v^-}{2},
# ```
# with $v^+$, and $v^-$ being the restrictions of $v\in V$ to the cells $T^+$, $T^-$ that share a generic interior facet in $\mathcal{F}_\Gamma$, and $n^+$, and $n^-$ are the facet outward unit normals from either the perspective of $T^+$ and $T^-$ respectively.
#
# With this notation, the weak form associated with the interior penalty formulation reads: find $u\in V$ such that $a(v,u) = b(v)$ for all $v\in V$. The bilinear $a(\cdot,\cdot)$ and linear form $b(\cdot)$ have contributions associated with the bulk of $\Omega$, and the boundary and interior facets $\mathcal{F}_{\partial\Omega}$, $\mathcal{F}_\Gamma$, namely 
# ``` math
# \begin{aligned}
# a(v,u) &= a_{\Omega}(v,u) + a_{\partial\Omega}(v,u) + a_{\Gamma}(v,u),\\
# b(v) &= b_{\Omega}(v) + b_{\partial\Omega}(v),
# \end{aligned}
# ```
# which are defined as
# ```math
# a_{\Omega}(v,u) \doteq \sum_{T\in\mathcal{T}} \int_{T} \nabla v \cdot \nabla u \ {\rm d}T, \quad b_{\Omega}(v) \doteq \int_{\Omega} v\ f \ {\rm d}\Omega,
# ```
# for the volume
# ```math
# \begin{aligned}
# a_{\partial\Omega}(v,u) &\doteq \sum_{F\in\mathcal{F}_{\partial\Omega}} \dfrac{\gamma}{|F|} \int_{F} v\ u \ {\rm d}F -  \sum_{F\in\mathcal{F}_{\partial\Omega}} \int_{F} v\ (\nabla u \cdot n)  \ {\rm d}F -  \sum_{F\in\mathcal{F}_{\partial\Omega}} \int_{F} (\nabla v \cdot n)\ u  \ {\rm d}F, \\
# b_{\partial\Omega} &\doteq \sum_{F\in\mathcal{F}_{\partial\Omega}} \dfrac{\gamma}{|F|} \int_{F} v\ g \ {\rm d}F  -  \sum_{F\in\mathcal{F}_{\partial\Omega}} \int_{F} (\nabla v \cdot n)\ g  \ {\rm d}F,
# \end{aligned}
# ```
# for the boundary facets and
# ```math
# a_{\Gamma}(v,u) \doteq \sum_{F\in\mathcal{F}_{\Gamma}} \dfrac{\gamma}{|F|} \int_{F} \lbrack\!\lbrack v\ n \rbrack\!\rbrack\cdot \lbrack\!\lbrack u\ n \rbrack\!\rbrack \ {\rm d}F -  \sum_{F\in\mathcal{F}_{\Gamma}} \int_{F} \lbrack\!\lbrack v\ n \rbrack\!\rbrack\cdot \{\! \!\{\nabla u\}\! \!\} \ {\rm d}F -  \sum_{F\in\mathcal{F}_{\Gamma}} \int_{F} \{\! \!\{\nabla v\}\! \!\}\cdot \lbrack\!\lbrack u\ n \rbrack\!\rbrack \ {\rm d}F,
# ```
# for the interior facets. In previous expressions, $|F|$ denotes the diameter of the face $F$ (in our Cartesian grid, this is equivalent to the characteristic mesh size $h$), and $\gamma$ is a stabilization parameter that should be chosen large enough such that the bilinear form $a(\cdot,\cdot)$ is stable and continuous. Here, we take $\gamma = p\ (p+1)$.
#
# ## 3D manufactured solution

using Gridap
import Gridap: ∇

u(x) = 3*x[1] + x[2] + 2*x[3]
∇u(x) = VectorValue(3.0,1.0,2.0)
∇(::typeof(u)) = ∇u
f(x) = 0.0
g(x) = u(x)

L = 1.0
limits = (0.0, L, 0.0, L, 0.0, L)
n = 4
model = CartesianDiscreteModel(domain=limits, partition=(n,n,n))

h = L / n

order = 3

fespace = FESpace(
  reffe=:Lagrangian,
  conformity = :L2,
  valuetype = Float64,
  model = model,
  order = order)

γ = order*(order+1)

V = TestFESpace(fespace)
U = TrialFESpace(fespace)

trian = Triangulation(model)
quad = CellQuadrature(trian,degree=2*order)

btrian = BoundaryTriangulation(model)
bquad = CellQuadrature(btrian,degree=2*order)

strian = SkeletonTriangulation(model)
squad = CellQuadrature(strian,degree=2*order)

# ![](../assets/t006_poisson_dg/skeleton_trian.png)

nb = NormalVector(btrian)
ns = NormalVector(strian)

writevtk(strian,"strian")

a_Ω(v,u) = inner(∇(v), ∇(u))
b_Ω(v) = inner(v,f)
t_Ω = AffineFETerm(a_Ω,b_Ω,trian,quad)

a_∂Ω(v,u) = (γ/h) * inner(v,u) - inner(v, ∇(u)*nb ) - inner(∇(v)*nb, u)
b_∂Ω(v) = (γ/h) * inner(v,g) - inner(∇(v)*nb, g)
t_∂Ω = AffineFETerm(a_∂Ω,b_∂Ω,btrian,bquad)

a_Γ(v,u) = (γ/h) * inner( jump(v*ns), jump(u*ns)) -
  inner( jump(v*ns), mean(∇(u)) ) - inner( mean(∇(v)), jump(u*ns) ) 
t_Γ = LinearFETerm(a_Γ,strian,squad)

op = LinearFEOperator(V,U,t_Ω,t_∂Ω,t_Γ)

uh = solve(op)

uh_Γ = restrict(uh,strian)

writevtk(strian,"jumps",
 cellfields=["jump_u"=>jump(uh_Γ), "jump_gradn_u"=> jump(∇(uh_Γ)*ns)])

# ![](../assets/t006_poisson_dg/jump_u.png)

e = u - uh

writevtk(trian,"trian",cellfields=["uh"=>uh,"e"=>e])

# ![](../assets/t006_poisson_dg/error.png)

l2(u) = inner(u,u)
h1(u) = a_Ω(u,u) + l2(u)

el2 = sqrt(sum( integrate(l2(e),trian,quad) ))
eh1 = sqrt(sum( integrate(h1(e),trian,quad) ))

#src @show el2
#src @show eh1

const k = 2*pi
u(x) = sin(k*x[1]) * x[2]
∇u(x) = VectorValue(k*cos(k*x[1])*x[2], sin(k*x[1]))
f(x) = (k^2)*sin(k*x[1])*x[2]
∇(::typeof(u)) = ∇u

function run(n,order)

  #Setup model
  L = 1.0
  h = L / n
  limits = (0.0, L, 0.0, L)
  partition = (n,n)
  model = CartesianDiscreteModel(domain=limits, partition=partition)

  #Setup FE spaces
  fespace = FESpace(
    reffe=:Lagrangian,
    conformity = :L2,
    valuetype = Float64,
    model = model,
    order = order)
  V = TestFESpace(fespace)
  U = TrialFESpace(fespace)

  #Setup integration meshes
  trian = Triangulation(model)
  btrian = BoundaryTriangulation(model)
  strian = SkeletonTriangulation(model)

  #Setup quadratures
  quad = CellQuadrature(trian,degree=2*order)
  squad = CellQuadrature(strian,degree=2*order)
  bquad = CellQuadrature(btrian,degree=2*order)

  #Setup normal vectors
  nb = NormalVector(btrian)
  ns = NormalVector(strian)

  #Setup weak form (volume)
  a_Ω(v,u) = inner(∇(v), ∇(u))
  b_Ω(v) = inner(v,f)
  
  #Setup weak form (boundary)
  γ = order*(order+1)
  a_∂Ω(v,u) = (γ/h) * inner(v,u) - inner(v, ∇(u)*nb ) - inner(∇(v)*nb, u)
  b_∂Ω(v) = (γ/h) * inner(v,g) - inner(∇(v)*nb, g)
  
  #Setup weak form (skeleton)
  a_Γ(v,u) = (γ/h) * inner( jump(v*ns), jump(u*ns)) -
    inner( jump(v*ns), mean(∇(u)) ) - inner( mean(∇(v)), jump(u*ns) ) 

  #Setup FE problem
  t_Ω = AffineFETerm(a_Ω,b_Ω,trian,quad)
  t_Γ = LinearFETerm(a_Γ,strian,squad)
  t_∂Ω = AffineFETerm(a_∂Ω,b_∂Ω,btrian,bquad)
  op = LinearFEOperator(V,U,t_Ω,t_∂Ω,t_Γ)

  #Solve
  uh = solve(op)

  #Measure discretization error
  e = u - uh
  l2(u) = inner(u,u)
  h1(u) = a_Ω(u,u) + l2(u)
  el2 = sqrt(sum( integrate(l2(e),trian,quad) ))
  eh1 = sqrt(sum( integrate(h1(e),trian,quad) ))

  (el2, eh1, h)
  
end

function conv_test(ns,order)

  el2s = Float64[]
  eh1s = Float64[]
  hs = Float64[]

  for n in ns
    @show n
    el2, eh1, h = run(n,order)
    push!(el2s,el2)
    push!(eh1s,eh1)
    push!(hs,h)
  end

  (el2s, eh1s, hs)

end


el2s, eh1s, hs = conv_test([8,16,32,64],3)

using Plots

plot(hs,[el2s eh1s],
    xaxis=:log, yaxis=:log,
    label=["L2" "H1"],
    shape=:auto,
    xlabel="h",ylabel="error norm")

#src savefig("conv.png")

function slope(hs,errors)
  x = log10.(hs)
  y = log10.(errors)
  linreg = hcat(fill!(similar(x), 1), x) \ y
  linreg[2]
end

slope(hs,el2s)

slope(hs,eh1s)

#md # ![](../assets/t006_poisson_dg/conv.png)


