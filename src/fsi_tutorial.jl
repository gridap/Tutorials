
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
u0 = 1.5
uf_in(x) = VectorValue( u0*4.0/0.1681*x[2]*(0.41-x[2]), 0.0 )
uf_0(x) = VectorValue( 0.0, 0.0 )
us_0(x) = VectorValue( 0.0, 0.0 )

# ### neumann
# ...
hN(x) = VectorValue( 0.0, 0.0 )
p_jump(x) = 0.0

# ## Body forces
f(x) = VectorValue( 0.0, 0.0 )
s(x) = VectorValue( 0.0, 0.0 )
g(x) = 0.0

# ## Discrete model
# Computational domain: elastic Flag
# *Turek, S., Hron, J., Madlik, M., Razzaq, M., Wobker, H., & Acker, J. F. (2011). Numerical simulation and benchmarking of a monolithic multigrid solver for fluid-structure interaction problems with application to hemodynamics. In Fluid Structure Interaction II (pp. 193-220). Springer, Berlin, Heidelberg.*
#model = CartesianDiscreteModel(domain, mesh)
model = DiscreteModelFromFile("models/elasticFlag.json")

# Triangulation of the full domain
trian = Triangulation(model)

# Solid & fluid triangulation & models
model_solid = DiscreteModel(model,"solid")
model_fluid = DiscreteModel(model,"fluid")
trian_solid = Triangulation(model_solid)
trian_fluid = Triangulation(model_fluid)

# ## FE Spaces
order = 2
# ### Case A: same FE space for fluid and solid velocities
Va = TestFESpace(
  model=model,
  valuetype=VectorValue{2,Float64},
  reffe=:Lagrangian,
  order=order,
  conformity =:H1,
  dirichlet_tags=["inlet", "noSlip", "cylinder", "fixed"])

# ### Case B: Different FE space for fluid and solid velocities
Vbf = TestFESpace(
    model=model_fluid,
    valuetype=VectorValue{2,Float64},
    reffe=:Lagrangian,
    order=order,
    conformity =:H1,
    dirichlet_tags=["inlet", "noSlip", "cylinder"])
Vbs = TestFESpace(
    model=model_solid,
    valuetype=VectorValue{2,Float64},
    reffe=:Lagrangian,
    order=order,
    conformity =:H1,
    dirichlet_tags=["fixed"])

Q = TestFESpace(
  model=model_fluid,
  valuetype=Float64,
  order=order-1,
  reffe=:Lagrangian,
  conformity=:C0)

Ua = TrialFESpace(Va,[uf_in, uf_0, uf_0, us_0])
Ubf = TrialFESpace(Vbf,[uf_in, uf_0, uf_0])
Ubs = TrialFESpace(Vbs,[us_0])
P = TrialFESpace(Q)

Ya = MultiFieldFESpace([Va,Q])
Yb = MultiFieldFESpace([Vbs,Vbf,Q])
Xa = MultiFieldFESpace([Ua,P])
Xb = MultiFieldFESpace([Ubs,Ubf,P])

# ## Numerical integration
# Interior quadratures:
degree = 2*order
quad = CellQuadrature(trian,degree)
quad_solid = CellQuadrature(trian_solid,degree)
quad_fluid = CellQuadrature(trian_fluid,degree)

# Boundary triangulations and quadratures:
bdegree = 2*order
trian_Γout = BoundaryTriangulation(model,"outlet")
quad_Γout = CellQuadrature(trian_Γout,bdegree)
n_Γout = get_normal_vector(trian_Γout)

# Interface triangulations and quadratures:
# This returns a SkeletonTriangulation whose normal vector
# goes outwards to the fluid domain.
idegree = 2*order
trian_Γfs = InterfaceTriangulation(model_fluid,model_solid)
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

# Fluid Cauchy stress tensor (deviatoric part)
@law σ_f(ε) = 2*μ_f*ε

# ### Weak form
# Case A
function a_solid_A(x,y)
  u,p = x
  v,q = y
  inner( ε(v), σ_s(ε(u)) )
end

function l_solid_A(y)
  v,q = y
  v*s
end

function a_fluid_A(x,y)
  u,p = x
  v,q = y
  inner( ε(v), σ_f(ε(u)) ) - (∇*v)*p + q*(∇*u)
end

function l_fluid_A(y)
  v,q = y
  v*f + q*g
end

function l_Γn_fluid_A(y)
  v,q = y
  v*hN
end

# Pressure drop at the interface
function l_Γ_A(y)
  v,q = y
  - mean(n_Γfs*v)*p_jump
end

# Case B
function a_solid_B(x,y)
  us,uf,p = x
  vs,vf,q = y
  inner( ε(vs), σ_s(ε(us)) )
end

function l_solid_B(y)
  vs,vf,q = y
  vs*s
end

function a_fluid_B(x,y)
  us,uf,p = x
  vs,vf,q = y
  inner( ε(vf), σ_f(ε(uf)) ) - (∇*vf)*p + q*(∇*uf)
end

function l_fluid_B(y)
  vs,vf,q = y
  vf*f + q*g
end

function l_Γn_fluid_B(y)
  vs,vf,q = y
  vf*hN
end

# Nitsche's method to enforce interface conditions
# See for instance: *Burman, Erik, and Miguel A. Fernández. "Stabilized explicit coupling for fluid–structure interaction using Nitsche's method." Comptes Rendus Mathematique 345.8 (2007): 467-472.*
const γ = 1.0
const h = 0.05
const χ = -1.0
function nitsche_Γ(x,y)
  us_Γ, uf_Γ, p_Γ = x
  vs_Γ, vf_Γ, q_Γ = y
  uf = jump(uf_Γ)
  p = jump(p_Γ)
  us = -jump(us_Γ)
  vf = jump(vf_Γ)
  q = jump(q_Γ)
  vs = -jump(vs_Γ)
  εuf = 0.5 * ( jump(∇(uf_Γ)) + jump(transpose(∇(uf_Γ))) )
  εvf = 0.5 * ( jump(∇(vf_Γ)) + jump(transpose(∇(vf_Γ))) )
  εus = 0.5 * ( -jump(∇(us_Γ)) + -jump(transpose(∇(us_Γ))) )
  εvs = 0.5 * ( -jump(∇(vs_Γ)) + -jump(transpose(∇(vs_Γ))) )

  # Penalty:
  penaltyTerms = (γ/h)*vf*uf - (γ/h)*vf*us - (γ/h)*vs*uf + (γ/h)*vs*us
  # Integration by parts terms:
  integrationByParts = ( vf*(p*n_Γfs) - vf*(σ_f(εuf)*n_Γfs) ) - ( vs*(p*n_Γfs) - vs*(σ_f(εuf)*n_Γfs) )
  # Symmetric terms:
  symmetricTerms =  ( -χ*q*(n_Γfs*uf) - χ*(σ_f(εvf)*n_Γfs)*uf ) - ( -χ*q*(n_Γfs*us) - χ*(σ_f(εvf)*n_Γfs)*us )

  penaltyTerms + integrationByParts + symmetricTerms
end

# Pressure drop at the interface
function l_Γ_B(y)
  vs,vf,q = y
  - n_Γfs*jump(vf)*p_jump
end

t_Ω_solid_A = AffineFETerm(a_solid_A,l_solid_A,trian_solid,quad_solid)
t_Ω_solid_B= AffineFETerm(a_solid_B,l_solid_B,trian_solid,quad_solid)
t_Ω_fluid_A = AffineFETerm(a_fluid_A,l_fluid_A,trian_fluid,quad_fluid)
t_Ω_fluid_B = AffineFETerm(a_fluid_B,l_fluid_B,trian_fluid,quad_fluid)
t_Γfs = AffineFETerm(nitsche_Γ,l_Γ_B,trian_Γfs,quad_Γfs)

t_Γn_fluid_A = FESource(l_Γn_fluid_A,trian_Γout,quad_Γout)
t_Γn_fluid_B = FESource(l_Γn_fluid_B,trian_Γout,quad_Γout)
t_Γ = FESource(l_Γ_A,trian_Γfs,quad_Γfs)

opA = AffineFEOperator(Xa,Ya,t_Ω_solid_A,t_Ω_fluid_A,t_Γn_fluid_A,t_Γ)
uhA, phA = solve(opA)
opB = AffineFEOperator(Xb,Yb,t_Ω_solid_B,t_Ω_fluid_B,t_Γn_fluid_B,t_Γfs)
uhsB, uhfB, phB = solve(opB)

# Visualization
phA_fluid = restrict(phA, trian_fluid)
phB_fluid = restrict(phB, trian_fluid)
uhfB_fluid = restrict(uhfB, trian_fluid)
uhsB_solid = restrict(uhsB, trian_solid)

writevtk(trian_fluid,"trian_fluid",cellfields=["phA"=>phA_fluid,"uhfB"=>uhfB_fluid])
writevtk(trian_solid,"trian_solid",cellfields=["uhsB"=>uhsB_solid])
writevtk(trian,"trian", cellfields=["uhA" => uhA, "phA"=> phA, "uhsB" => uhsB, "uhfB" => uhfB, "phB" => phB])

# Quantities of Interest
trian_ΓS = BoundaryTriangulation(model,["cylinder","interface"])
quad_ΓS = CellQuadrature(trian_ΓS,bdegree)
n_ΓS = get_normal_vector(trian_ΓS)
uh_ΓS = restrict(uhfB_fluid,trian_ΓS)
ph_ΓS = restrict(phB_fluid,trian_ΓS)
FD, FL = sum( integrate( (σ_f(ε(uh_ΓS))*n_ΓS - ph_ΓS*n_ΓS), trian_ΓS, quad_ΓS ) )
println("Drag force: ", FD)
println("Lift force: ", FL)
end # module