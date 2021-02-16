module KirchhoffPlate

using Gridap
using GridapODEs
using GridapODEs.TransientFETools
using GridapODEs.ODETools
using WriteVTK

n = 40
domain = (0,1,0,1)
partition = (n,n)
model = CartesianDiscreteModel(domain,partition)

order = 2
reffe = ReferenceFE(lagrangian,Float64,order)
V = TestFESpace(model,reffe,conformity=:H1,dirichlet_tags="boundary")
U = TransientTrialFESpace(V)

Ω = Triangulation(model)
Λ = SkeletonTriangulation(model)
dΩ = Measure(Ω,2*order)
dΛ = Measure(Λ,2*order)
nΛ = get_normal_vector(Λ)

q(t::Real) = 0.0
u(x) = 0.0
v(x) = x[1]*(1.0-x[1])*x[2]*(1.0-x[2])
a(x) = -x[1]*(1.0-x[1])*x[2]*(1.0-x[2])

const γ = 1.0
const h = 1/n
m(t,utt,v) = ∫(utt⋅v)dΩ
c(t,ut,v) = ∫(0.0*ut⋅v)dΩ
a(t,u,v) = ∫(Δ(u)*Δ(v))dΩ + ∫( - jump(nΛ⋅∇(u))*mean(Δ(v)) - mean(Δ(u))*jump(nΛ⋅∇(v)) + γ/h*jump(nΛ⋅∇(u))*jump(nΛ⋅∇(v)) )dΛ
l(t,v) = ∫(q(t)*v)dΩ

op = TransientAffineFEOperator(m,c,a,l,U,V)

t0 = 0.0
tF = π/10
dt = π/200

u₀ = interpolate_everywhere(u,U(0.0))
v₀ = interpolate_everywhere(v,U(0.0))
a₀ = interpolate_everywhere(a,U(0.0))

ls = LUSolver()
odes = Newmark(ls,dt,0.5,0.25)
solver = TransientFESolver(odes)

sol = solve(solver,op,u₀,v₀,a₀,t0,tF)

outfiles = paraview_collection("tmp", append=false) do pvd
  for (uₙ,tₙ) in sol
    pvd[tₙ] = createvtk(Ω,"tmp_$tₙ.vtu",cellfields = ["u" => uₙ]) 
  end
end

end