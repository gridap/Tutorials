module FSI_failing_test
using Gridap
using Gridap.Arrays
using Gridap.FESpaces
using Gridap.Geometry
import Gridap: ∇, ε
using LinearAlgebra: tr

uf_in(x) = VectorValue( (1+x[2])*(1-x[2]), 0.0 )
uf_0(x) = VectorValue( 0.0, 0.0 )
us_0(x) = VectorValue( 0.0, 0.0 )
hN(x) = VectorValue( 0.0, 0.0 )
p_jump(x) = 0.0
f(x) = VectorValue( 0.0, 0.0 )
s(x) = VectorValue( 0.0, 0.0 )
g(x) = 0.0

mesh = (20,10)
domain = (0.0,4.0,-1.0,1.0)
order = 2
model = CartesianDiscreteModel(domain, mesh)
trian = Triangulation(model)
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"inlet",[7])
add_tag_from_tags!(labels,"noSlip",[1,2,3,4,5,6])
add_tag_from_tags!(labels,"outlet",[8])

const width = 0.5
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
model_solid = DiscreteModel(model, cell_to_is_solid)
model_fluid = DiscreteModel(model, cell_to_is_fluid)

Vbf = TestFESpace(
    model=model_fluid,
    valuetype=VectorValue{2,Float64},
    reffe=:QLagrangian,
    order=order,
    conformity =:H1,
    dirichlet_tags=["inlet", "noSlip"])
Vbs = TestFESpace(
    model=model_solid,
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

  Ubf = TrialFESpace(Vbf,[uf_in, uf_0])
  Ubs = TrialFESpace(Vbs,[us_0])
  P = TrialFESpace(Q)

  Yb = MultiFieldFESpace([Vbs,Vbf,Q])
  Xb = MultiFieldFESpace([Ubs,Ubf,P])

  idegree = 2*order
  trian_Γfs = InterfaceTriangulation(model,cell_to_is_fluid)
  n_Γfs = get_normal_vector(trian_Γfs)
  n_Γsf = - n_Γfs
  quad_Γfs = CellQuadrature(trian_Γfs,idegree)

  const μ_f = 1.0
  @law σ_f(ε) = 2*μ_f*ε
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
    # εuf = 0.5 * ( jump(∇(uf_Γ)) + jump(transpose(∇(uf_Γ))) )
    # εvf = 0.5 * ( jump(∇(vf_Γ)) + jump(transpose(∇(vf_Γ))) )
    εuf = jump(ε(uf_Γ))
    εvf = jump(ε(vf_Γ))

    # Penalty:
    penaltyTerms = (γ/h)*vf*uf - (γ/h)*vf*us - (γ/h)*vs*uf + (γ/h)*vs*us
    # Integration by parts terms:
    integrationByParts = ( vf*(p*n_Γfs) - vf*(σ_f(εuf)*n_Γfs) ) - ( vs*(p*n_Γfs) - vs*(σ_f(εuf)*n_Γfs) )
    # Symmetric terms:
    symmetricTerms =  ( -χ*q*(n_Γfs*uf) - χ*(σ_f(εvf)*n_Γfs)*uf ) - ( -χ*q*(n_Γfs*us) - χ*(σ_f(εvf)*n_Γfs)*us )

    penaltyTerms + integrationByParts + symmetricTerms
  end

  function l_Γ_B(y)
    vs,vf,q = y
    - n_Γfs*jump(vf)*p_jump
  end

  t_Γfs = AffineFETerm(nitsche_Γ,l_Γ_B,trian_Γfs,quad_Γfs)
  #t_Γfs = FESource(l_Γ_B,trian_Γfs,quad_Γfs)
  opB = AffineFEOperator(Xb,Yb,t_Γfs)

end
