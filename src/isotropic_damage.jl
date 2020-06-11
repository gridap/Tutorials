
#md # !!! note
#
#     This tutorial is under construction, but the code below is already functional.
#

using Gridap
using LinearAlgebra

# ## Model definition
#
# Elastic branch
const E = 3.0e10 # Pa
const ν = 0.3 # dim-less
const λ = (E*ν)/((1+ν)*(1-2*ν))
const μ = E/(2*(1+ν))
@law σe(ε) = λ*tr(ε)*one(ε) + 2*μ*ε # Pa
τ(ε) = sqrt(ε ⊙ σe(ε)) # Pa^(1/2)

# Damage
const σ_u = 4.0e5 # Pa
const r_0 = σ_u / sqrt(E) # Pa^(1/2)
const H = 0.5 # dim-less

function d(r)
  1 - q(r)/r
end

function q(r)
  r_0 + H*(r-r_0)
end

# Update of the state variables
@law function update(ε_in,r_in,d_in)
  τ_in = τ(ε_in)
  if τ_in <= r_in
    r_out = r_in
    d_out = d_in
    damaged = false
  else
    r_out = τ_in
    d_out = d(r_out)
    damaged = true
  end
  damaged, r_out, d_out
end

# Constitutive law and its linearization

@law function σ(ε_in,r_in,d_in)

  _, _, d_out = update(ε_in,r_in,d_in)

  (1-d_out)*σe(ε_in)

end

@law function dσ(dε_in,ε_in,state)

  damaged, r_out, d_out = state

  if ! damaged
    return (1-d_out)*σe(dε_in)

  else
    c_inc = ((q(r_out) - H*r_out)*(σe(ε_in) ⊙ dε_in))/(r_out^3)
    return (1-d_out)*σe(dε_in) - c_inc*σe(ε_in)

  end
end

# max dead load
const b_max = VectorValue(0.0,0.0,-(9.81*2.5e3))

# ## L2 projection
# form Gauss points to a Lagrangian piece-wise discontinuous space
function project(q,trian,quad,order)

  a(u,v) = u*v
  l(v) = v*q
  t_Ω = AffineFETerm(a,l,trian,quad)

  V = TestFESpace(
    reffe=:Lagrangian, valuetype=Float64, order=order,
    triangulation=trian, conformity=:L2)

  U = TrialFESpace(V)
  op = AffineFEOperator(U,V,t_Ω)
  qh = solve(op)
  qh
end

# ## Main function
function main(;n,nsteps)

  r = 12
  domain = (0,r,0,1,0,1)
  partition = (r*n,n,n)
  model = CartesianDiscreteModel(domain,partition)

  labeling = get_face_labeling(model)
  add_tag_from_tags!(labeling,"supportA",[1,3,5,7,13,15,17,19,25])
  add_tag_from_tags!(labeling,"supportB",[2,4,6,8,14,16,18,20,26])
  add_tag_from_tags!(labeling,"supports",["supportA","supportB"])

  order = 1

  V = TestFESpace(
    reffe=:Lagrangian, valuetype=VectorValue{3,Float64}, order=order,
    model=model,
    labels = labeling,
    dirichlet_tags=["supports"])

  U = TrialFESpace(V)

  trian = Triangulation(model)
  degree = 2*order
  quad = CellQuadrature(trian,degree)

  r = QPointCellField(r_0,trian,quad)
  d = QPointCellField(0.0,trian,quad)

  nls = NLSolver(show_trace=true, method=:newton)
  solver = FESolver(nls)

  function step(uh_in,factor)

    b = factor*b_max
    res(u,v) = ( ε(v) ⊙ σ(ε(u),r,d) ) - v⋅b
    function jac(u,du,v)
      state = update(ε(u),r,d)
      ε(v) ⊙ dσ(ε(du),ε(u),state)
    end
    t_Ω = FETerm(res,jac,trian,quad)
    op = FEOperator(U,V,t_Ω)

    uh_out, = solve!(uh_in,solver,op)

    update_state_variables!(quad,update,ε(uh_out),r,d)

    uh_out
  end

  factors = collect(1:nsteps)*(1/nsteps)
  uh = zero(V)

  for (istep,factor) in enumerate(factors)

    println("\n+++ Solving for load factor $factor in step $istep of $nsteps +++\n")

    uh = step(uh,factor)
    dh = project(d,trian,quad,order)
    rh = project(r,trian,quad,order)

    writevtk(
      trian,"results_$(lpad(istep,3,'0'))",
      cellfields=["uh"=>uh,"epsi"=>ε(uh),"damage"=>dh,
                  "threshold"=>rh,"sigma_elast"=>σe(ε(uh))])

  end

end

# Run!
main(n=6,nsteps=20)

# ## Results

# Animation of the load history using for `main(n=8,nsteps=30)`
# ![](../assets/isotropic_damage/damage.gif)

