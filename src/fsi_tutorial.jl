
# # Tutorial 9: Fluid-structure interaction
#
#md # [![](https://mybinder.org/badge_logo.svg)](@__BINDER_ROOT_URL__/notebooks/fsi_tutorial.ipynb)
#md # [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/notebooks/fsi_tutorial.ipynb)
#
# In this tutorial, we will learn
#
#    -  Bla, bla
#    -  Bla
#
#
# ## Problem statement
#  Let $\Gamma_{\rm FS}$ be the interface between a fluid domain $\Omega_{\rm F}$ and a solid domain $\Omega_{\rm S}$. We denote by $\Gamma_{\rm F,D}$ and $\Gamma_{\rm F,N}$ the fluid boundaries with Dirichlet and Neumann conditions, respectively.
# The Fluid-Structure Interaction (FSI) problem reads:
#
# find $ u_{\rm F} $, $ p_{\rm F} $ and $ u_{\rm S} $ such that
# ```math
# \left\lbrace
# \begin{aligned}
# -\nabla\cdot\boldsymbol{\sigma}_{\rm F} = f &\text{ in }\Omega_{\rm F},\\
# \nabla\cdot u_{\rm F} = 0 &\text{ in } \Omega_{\rm F},\\
# -\nabla\cdot\boldsymbol{\sigma}_{\rm S} = s &\text{ in }\Omega_{\rm S},\\
# \end{aligned}
# \right.
# ```
#
# satisfying the Dirichlet and Neumann boundary conditions
# ```math
# \left\lbrace
# \begin{aligned}
# u_{\rm F} = g &\text{ on } \Gamma_{\rm F,D},\\
# \boldsymbol{\sigma}_{\rm F}\cdot n_{\rm F} = 0 &\text{ on } \Gamma_{\rm F,N},\\
# \end{aligned}
# \right.
# ```
#
# and the kinematic and dynamic conditions at the fluid-solid interface
# ```math
# \left\lbrace
# \begin{aligned}
# u_{\rm F} = u_{\rm S} &\text{ on } \Gamma_{\rm FS},\\
# \boldsymbol{\sigma}_{\rm F}\cdot n_{\rm F} + \boldsymbol{\sigma}_{\rm S}\cdot n_{\rm S} = 0 &\text{ on } \Gamma_{\rm FS}.\\
# \end{aligned}
# \right.
# ```
#
# Where $\boldsymbol{\sigma}_{\rm F}(u_{\rm F},p_{\rm F})=2\mu_{\rm F}\boldsymbol{\varepsilon}(u_{\rm F}) - p_{\rm F}\mathbf{I}$ and $\boldsymbol{\sigma}_{\rm S}(u_{\rm S})=2\mu_{\rm S}\boldsymbol{\varepsilon}(u_{\rm S}) +\lambda_{\rm S}tr(\boldsymbol{\varepsilon}(u_{\rm S}))\mathbf{I}$.
#
# In this tutorial we consider a square computational domain $\Omega \doteq (-1,1)^2$, with the associated FE triangulation $\mathcal{T}$. The solid domain is composed by the union of cells whose centroid is inside a circle of radius R=0.4 and center $\mathbf{x}_o=(0,0)$, that is
# ```math
# \Omega_{\rm S}\doteq \bigcup_{T\in\mathcal{T}_{\rm S}}T, \quad \mbox{with }\quad\mathcal{T}_{\rm S}\doteq\{T\in\mathcal{T}\vert\|\mathbf{x}_T-\mathbf{x}_o\|<R\}.
# ```
#
# Then, the fluid domain will be defined by the remaining part, i.e. $\Omega_{\rm F}=\Omega\backslash\Omega_{\rm S}$, with $\Omega_{\rm F}\cap\Omega_{\rm S}=\Gamma_{\rm FS}$.
#
# ## Numerical scheme
#

# ## Setup environment

module FSITest

using Test
using Gridap
using Gridap.Arrays
using Gridap.FESpaces
import Gridap: ∇, ε
using LinearAlgebra: tr
using Gridap.Geometry


# ## Definition of the Boundary conditions

# ### Dirichlet
# ```math
# \left\lbrace
# \begin{aligned}
# u_{\rm F,in}(x,y) = [(1+y)(1-y), 0]\quad\mbox{on }\Gamma_{\rm F,D_{in}}
# u_{\rm F,0}(x,y) = [0, 0]\quad\mbox{on }\Gamma_{\rm F,D_{0}}
# u_{\rm S,0}(x,y) = [0, 0]\quad\mbox{on }\Gamma_{\rm S,D_{0}}
# ```
uf_in(x) = VectorValue( (1+x[2])*(1-x[2]), 0.0 )
uf_0(x) = VectorValue( 0.0, 0.0 )
us_0(x) = VectorValue( 0.0, 0.0 )

# ### neumann
# ...
h(x) = VectorValue( 0.0, 0.0 )
p_jump(x) = 0.0

# ## Body forces
f(x) = VectorValue( 0.0, 0.0 )
s(x) = VectorValue( 0.0, 0.0 )
g(x) = 0.0

# ## Discrete model
# Computational domain: channel of size 4x2
mesh = (80,40)
domain = (0.0,4.0,-1.0,1.0)
order = 2
model = CartesianDiscreteModel(domain, mesh)

# Triangulation of the full domain
trian = Triangulation(model)

# Boundary conditions on the full domain
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"inlet",[7])
add_tag_from_tags!(labels,"noSlip",[1,2,3,4,5,6])
add_tag_from_tags!(labels,"outlet",[8])

# Solid & fluid triangulation
const width = 0.2
const height = 1.2
const x0 = 2.0
function is_in(coords)
  n = length(coords)
  x = (1/n)*sum(coords)
  box_min_x = x0 - width/2.0
  box_min_y = -1.0
  box_max_x = x0 + width/2.0
  box_max_y = height - 1.0
  (x[1] > box_min_x) && (x[2] > box_min_y) && (x[1] < box_max_x) && (x[2] < box_max_y)
end
cell_to_coods = get_cell_coordinates(trian)
cell_to_is_solid = collect1d(apply(is_in,cell_to_coods))
cell_to_is_fluid = Vector{Bool}(.! cell_to_is_solid)
trian_solid = RestrictedTriangulation(trian, cell_to_is_solid)
trian_fluid = RestrictedTriangulation(trian, cell_to_is_fluid)

# ## FE Spaces
# ### Case A: same FE space for fluid and solid velocities
Va = TestFESpace(
  model=model,
  valuetype=VectorValue{2,Float64},
  reffe=:QLagrangian,
  order=order,
  conformity =:H1,
  dirichlet_tags=["inlet", "noSlip"])

# ### Case B: Different FE space for fluid and solid velocities
Vbf = TestFESpace(
    triangulation=trian_fluid,
    valuetype=VectorValue{2,Float64},
    reffe=:QLagrangian,
    order=order,
    conformity =:H1,
    dirichlet_tags=["inlet", "noSlip"])
Vbs = TestFESpace(
    triangulation=trian_solid,
    valuetype=VectorValue{2,Float64},
    reffe=:QLagrangian,
    order=order,
    conformity =:H1,
    dirichlet_tags=["noSlip"])

Q = TestFESpace(
  triangulation=trian_fluid,
  valuetype=Float64,
  order=order-1,
  reffe=:PLagrangian,
  conformity=:L2)


Ua = TrialFESpace(Va,[uf_in, uf_0])
Ubf = TrialFESpace(Vbf,[uf_in, uf_0])
Ubs = TrialFESpace(Vbs,[us_0])
P = TrialFESpace(Q)

Ya = MultiFieldFESpace([Va,Q])
Yb = MultiFieldFESpace([Vbs,Vbf,Q])
Xa = MultiFieldFESpace([Ua,P])
Xb = MultiFieldFESpace([Ubs,Ubf,P])

# ## Numerical integration
# Interior quadratures:
order = 2
degree = 2*order
quad = CellQuadrature(trian,degree)
quad_solid = CellQuadrature(trian_solid,degree)
quad_fluid = CellQuadrature(trian_fluid,degree)

# Boundary triangulations and quadratures:
bdegree = 2*order
trian_Γout = BoundaryTriangulation(model,labels,"outlet")
quad_Γout = CellQuadrature(trian_Γout,bdegree)
n_Γout = get_normal_vector(trian_Γout)

# Interface triangulations and quadratures:
# This returns a SkeletonTriangulation whose normal vector
# goes outwards to the fluid domain.
idegree = 2*order
trian_Γfs = InterfaceTriangulation(model,cell_to_is_fluid)
n_Γfs = get_normal_vector(trian_Γfs)
n_Γsf = - n_Γfs
quad_Γfs = CellQuadrature(trian_Γfs,idegree)

# FE problem
using LinearAlgebra: tr

# Elasticity properties
function lame_parameters(E,ν)
  λ = (E*ν)/((1+ν)*(1-2*ν))
  μ = E/(2*(1+ν))
  (λ, μ)
end

const E_s = 1.0
const ν_s = 0.33
const (λ_s,μ_s) = lame_parameters(E_s,ν_s)

# Solid Cauchy stress tensor
@law σ_s(ε) = λ_s*tr(ε)*one(ε) + 2*μ_s*ε

const E_f = 1.0
const ν_f = 0.5
const (λ_f,μ_f) = lame_parameters(E_f,ν_f)

# Fluid Cauchy stress tensor
@law σ_f(ε) = 2*μ_f*ε

# ### Weak conform
function a_solid(x,y)
  u,p = x
  v,q = y
  inner( ε(v), σ_s(ε(u)) )
end

function l_solid(y)
  v,q = y
  v*s
end

function a_fluid(x,y)
  u,p = x
  v,q = y
  inner( ε(v), σ_f(ε(u)) ) - (∇*v)*p + q*(∇*u)
end

function l_fluid(y)
  v,q = y
  v*f + q*g
end

function l_Γn_fluid(y)
  v,q = y
  v*h
end

# Pressure drop at the interface
function l_Γ(y)
  v,q = y
  - mean(n_Γ*v)*p_jump
end

# Nitsche's method to enforce interface conditions
function nitsche_Γ(x,y)
  us_Γ, uf_Γ, p_Γ = x
  vs_Γ, vf_Γ, q_Γ = y
  uf = uf_Γ.left
  p = p_Γ.left
  us = us_Γ.right
  vf = vf_Γ.left
  q = q_Γ.left
  vs = vs_Γ.right
  # Penalty:
  penaltyTerms = (γ/h) * (vf - vs) * (uf - us)
  # Integration by parts terms:
  integrationByParts = (vf - vs) * (p*n_Γfs - n_Γfs*ε(uf) - n_Γsf*ε(us))
  # Symmetric terms:
  symmetricTerms = (q*n_Γfs - n_Γfs*ε(vf) - n_Γsf*ε(vs)) * (uf - us)
  # (γ/h)*vf*uf - vf*(n_Γ*ε(uf)) - (n_Γ*ε(vf))*uf + (p*n_Γ)*vf + (q*n_Γ)*uf
  #    - (γ/h)*vf*us + (n_Γ*ε(vf))*us - (q*n_Γ)*us
  #    + (γ/h)*vs*us + vs*(n_Γ*ε(us)) + (n_Γ*ε(vs))*us
  #    - (γ/h)*vf*uf - (n_Γ*ε(vf))*uf + (q*n_Γ)*uf
  penaltyTerms + integrationByParts + symmetricTerms
end

t_Ω_solid = AffineFETerm(a_solid,l_solid,trian_solid,quad_solid)
t_Ω_fluid = AffineFETerm(a_fluid,l_fluid,trian_fluid,quad_fluid)
t_Γfs = AffineFETerm(nitsche_Γ,l_Γ,trian_Γfs,quad_Γfs)
t_Γn_fluid = FESource(l_Γn_fluid,trian_Γout,quad_Γout)
t_Γ = FESource(l_Γ,trian_Γ,quad_Γ)

opA = AffineFEOperator(Xa,Ya,t_Ω_solid,t_Ω_fluid,t_Γn_fluid,t_Γ)
uhA, phA = solve(opA)
opB = AffineFEOperator(Xb,Yb,t_Ω_solid,t_Ω_fluid,t_Γn_fluid,t_Γfs)
uhA, phA = solve(opA)
uhsB, uhfB, phB = solve(opA)

# Visualization
phA_fluid = restrict(phA, trian_fluid)
phB_fluid = restrict(phB, trian_fluid)
uhfB_fluid = restrict(uhfB, trian_fluid)
uhsB_fluid = restrict(uhsB, trian_solid)

#eu = u - uh
#ep = p - ph
#ep_fluid = p - ph_fluid

writevtk(trian_fluid,"trian_fluid",cellfields=["phA"=>phA_fluid,"uhfB"=>uhfB_fluid])
writevtk(trian_solid,"trian_solid",cellfields=["uhsB"=>uhsB_solid])
#
writevtk(trian,"trian", cellfields=["uhA" => uhA, "phA"=> phA, "uhsB" => uhsB, "uhfB" => uhfB, "phB" => phB])

# Errors

# l2(v) = v*v
# h1(v) = v*v + inner(∇(v),∇(v))
#
# eu_l2 = sqrt(sum(integrate(l2(eu),trian,quad)))
# eu_h1 = sqrt(sum(integrate(h1(eu),trian,quad)))
# ep_l2 = sqrt(sum(integrate(l2(ep_fluid),trian_fluid,quad_fluid)))
#
# tol = 1.0e-9
# @test eu_l2 < tol
# @test eu_h1 < tol
# @test ep_l2 < tol

end # module
