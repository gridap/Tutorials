module transient_fsi

using Gridap
using Gridap.Arrays
using Gridap.Geometry
using GridapODEs.ODETools
using GridapODEs.TransientFETools
using LinearAlgebra: tr, inv, det
using LineSearches: BackTracking
using WriteVTK
import GridapODEs.TransientFETools: ∂t

# ## Problem setting
# ### Problem parameters
const Um = 1.0
const H = 0.41
const ⌀ = 0.1
const ρ = 1.0
const t0 = 0.0
const Re = 100.0

# ### Boundary conditions
u_in(x, t) = VectorValue(1.5 * Um * x[2] * (H - x[2]) / ((H / 2)^2), 0.0)
u_noSlip(x, t) = VectorValue(0.0, 0.0)
u_in(t::Real) = x -> u_in(x, t)
u_noSlip(t::Real) = x -> u_noSlip(x, t)
∂tu_in(t) = x -> VectorValue(0.0, 0.0)
∂tu_in(x, t) = ∂tu_in(t)(x)
∂t(::typeof(u_in)) = ∂tu_in
∂t(::typeof(u_noSlip)) = ∂tu_in

# ### Material properties
function lame_parameters(E,ν)
	λ = (E*ν)/((1+ν)*(1-2*ν))
	μ = E/(2*(1+ν))
	(λ, μ)
end
(λ_s,μ_s) = lame_parameters(1.4e6, 0.4)
ρ_s = 1.0e4
ρ_f = 1.0e3
μ_f = ρ_f * Um * ⌀ / Re
α_u = 1.0e-5

# ## Geometry
# ### Discrete model
#model = DiscreteModelFromFile("../runs/elasticFlag_coarse.json")
model = DiscreteModelFromFile("../models/elasticFlag.json")
model_solid = DiscreteModel(model,tags="solid")
model_fluid = DiscreteModel(model,tags="fluid")
writevtk(model,"model")

# ## Triangulations and integration Measures
Ω = Triangulation(model)
Ωs = Triangulation(model_solid)
Ωf = Triangulation(model_fluid)
Γi = BoundaryTriangulation(model_fluid,tags="interface")
Λf = SkeletonTriangulation(model_fluid)
order = 2
degree = 2*order
bdegree = 2*order
dΩs = Measure(Ωs,degree)
dΩf = Measure(Ωf,degree)
dΓi = Measure(Γi,bdegree)
dΛf = Measure(Λf,bdegree)
nΓi = get_normal_vector(Γi)
nΛs = get_normal_vector(Λf)

# ## Lagrangian / ALE map related quantities
F(∇u) = ∇u + one(∇u)
J(∇u) = det(F(∇u))
Finv(∇u) = inv(F(∇u))
FinvT(∇u) = (Finv(∇u)')
dF(∇du) = ∇du
dJ(∇u,∇du) = J(F(∇u))*tr(inv(F(∇u))⋅dF(∇du))
dFinv(∇u,∇du) = -Finv(∇u) ⋅ dF(∇du) ⋅ Finv(∇u)
dFinvT(∇u,∇du) = (dFinv(∇u,∇du)')

# ## Constitutive laws
E(∇u) = 0.5 * ((F(∇u)')⋅F(∇u) - one(F(∇u)))
S(∇u) = 2*μ_s*E(∇u) + λ_s*tr(E(∇u))*one(E(∇u))
σ_dev(∇v,Finv) = μ_f*(∇v⋅Finv + (Finv')⋅(∇v'))
conv(c,∇v) = (∇v') ⋅ c
dE(∇u,∇du) = 0.5 * ((dF(∇du)')⋅F(∇u) + (F(∇u)')⋅dF(∇du))
dS(∇u,∇du) = 2*μ_s*dE(∇u,∇du) + λ_s*tr(dE(∇u,∇du))*one(E(∇u))
dconv(dc,∇dv,c,∇v) = conv(c,∇dv) + conv(dc,∇v)

# ## FE Spaces
# ### Reference Finite Elements
refFEᵤ = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
refFEᵥ = refFEᵤ
refFEₚ = ReferenceFE(lagrangian,Float64,order-1)

# ### Test FE spaces
Vuf = TestFESpace(model_fluid,refFEᵤ,conformity =:H1,dirichlet_tags=["inlet", "noSlip", "cylinder","interface","outlet"])
Vvf = TestFESpace(model_fluid,refFEᵥ,conformity =:H1,dirichlet_tags=["inlet", "noSlip", "cylinder","interface"])
Vu = TestFESpace(model,refFEᵤ,conformity =:H1,dirichlet_tags=["inlet", "noSlip", "cylinder","fixed","outlet"])
Vv = TestFESpace(model,refFEᵥ,conformity =:H1,dirichlet_tags=["inlet", "noSlip", "cylinder","fixed"])
Qf = TestFESpace(model_fluid,refFEₚ,conformity=:C0)

# ### Trial FE Spaces
Uuf = TrialFESpace(Vuf,[u_noSlip(0.0), u_noSlip(0.0), u_noSlip(0.0), u_noSlip(0.0), u_noSlip(0.0)])
Uvf = TrialFESpace(Vvf,[u_in(0.0), u_noSlip(0.0), u_noSlip(0.0), u_noSlip(0.0)])
Uu = TransientTrialFESpace(Vu,[u_noSlip, u_noSlip, u_noSlip, u_noSlip, u_noSlip])
Uv = TransientTrialFESpace(Vv,[u_in, u_noSlip, u_noSlip, u_noSlip])
Pf = TrialFESpace(Qf)

# ### Multifield FE Spaces
Yf = MultiFieldFESpace([Vuf,Vvf,Qf])
Xf = MultiFieldFESpace([Uuf,Uvf,Pf])
Y = MultiFieldFESpace([Vu,Vv,Qf])
X = TransientMultiFieldFESpace([Uu,Uv,Pf])

# ## Stokes FE Operator
# ### Weak Form
a_ST((u,v,p),(ϕ,φ,q)) = ∫( ∇(u)⊙∇(ϕ) + 2*μ_f*ε(v)⊙ε(φ) + p*(∇⋅φ) + (∇⋅v)*q )dΩf
l_ST((ϕ,φ,q)) = 0.0

# ### Affine FE operator
op_ST = AffineFEOperator(a_ST,l_ST,Xf,Yf)

# ## FSI FE Operator
# ### Auxiliar variables
using Gridap.CellData
const γ = 1.0
dim = num_cell_dims(Λf)
hΛf = get_array(∫(1)dΛf)
αΛf = CellField( lazy_map(h->γ*μ_f/(h.^dim),hΛf), Λf)

# ### Residual
function res(t,(u,v,p),(ut,vt,pt),(ϕ,φ,q))
	∫( α_u*Δ(u)*Δ(ϕ) + 
		 ( (J∘∇(u)) * ρ_f * vt ) ⋅ φ + 
		 ( (J∘∇(u)) * ρ_f * (conv∘((Finv∘∇(u))⋅(v-ut), ∇(v)) )) ⋅ φ +
		 ( (J∘∇(u)) * (σ_dev∘(∇(v),Finv∘∇(u))) ⋅ (FinvT∘∇(u)) ) ⊙ ∇(φ) + 
		 ( (J∘∇(u)) * p * tr(FinvT∘∇(u)) ) * (∇⋅φ) +
		 ( (J∘∇(u)) * (∇(v)⊙(FinvT∘∇(u))) ) * q )dΩf +
	∫( - mean(Δ(u))*jump(∇(ϕ)⋅nΛf) - jump(∇(u)⋅nΛf)*mean(Δ(ϕ)) + αΛf*jump(∇(u)⋅nΛf)*jump(∇(ϕ)⋅nΛf) )dΛf +
	∫( (ut-v)⋅ϕ + 0.0*(u⋅ϕ) + ρ_s*(vt⋅φ) + 0.0*(φ⋅v) + ((F∘∇(u))⋅(S∘∇(u))) ⊙ ∇(φ) )dΩs
end

# ### Spatial Jacobian
function jac(t,(u,v,p),(ut,vt,pt),(du,dv,dp),(ϕ,φ,q))
	∫( α_u*Δ(du)*Δ(ϕ) + 
		 ( (dJ∘(∇(u),∇(du))) * ρ_f * vt ) ⋅ φ + 
		 ( (dJ∘(∇(u),∇(du))) * ρ_f * (conv∘((Finv∘∇(u))⋅(v-ut), ∇(v)) )) ⋅ φ +
		 ( (J∘∇(u)) * ρ_f * (conv∘((dFinv∘(∇(u),∇(du)))⋅(v-ut), ∇(v)) )) ⋅ φ +
		 ( (J∘∇(u)) * ρ_f * (dconv∘((Finv∘∇(u))⋅dv, ∇(dv), (Finv∘∇(u))⋅(v-ut), ∇(v)) )) ⋅ φ +
		 ( (dJ∘(∇(u),∇(du))) * (σ_dev∘(∇(v),Finv∘∇(u))) ⋅ (FinvT∘∇(u)) ) ⊙ ∇(φ) + 
		 ( (J∘∇(u)) * (σ_dev∘(∇(v),dFinv∘(∇(u),∇(du)))) ⋅ (FinvT∘∇(u)) ) ⊙ ∇(φ) + 
		 ( (J∘∇(u)) * (σ_dev∘(∇(v),Finv∘∇(u))) ⋅ (dFinvT∘(∇(u),∇(du))) ) ⊙ ∇(φ) + 
		 ( (J∘∇(u)) * (σ_dev∘(∇(dv),Finv∘∇(u))) ⋅ (FinvT∘∇(u)) ) ⊙ ∇(φ) + 
		 ( (dJ∘(∇(u),∇(du))) * p * tr(FinvT∘∇(u)) ) * (∇⋅φ) +
		 ( (J∘∇(u)) * p * tr(dFinvT∘(∇(u),∇(du))) ) * (∇⋅φ) +
		 ( (J∘∇(u)) * dp * tr(FinvT∘∇(u)) ) * (∇⋅φ) +
		 ( (dJ∘(∇(u),∇(du))) * (∇(v)⊙(FinvT∘∇(u))) ) * q  +
		 ( (J∘∇(u)) * (∇(v)⊙(dFinvT∘(∇(u),∇(du)))) ) * q  +
		 ( (J∘∇(u)) * (∇(dv)⊙(FinvT∘∇(u))) ) * q )dΩf +
	∫( - mean(Δ(du))*jump(∇(ϕ)⋅nΛf) - jump(∇(du)⋅nΛf)*mean(Δ(ϕ)) + αΛf*jump(∇(du)⋅nΛf)*jump(∇(ϕ)⋅nΛf) )dΛf +
	∫( -dv⋅ϕ + 0.0*(du⋅ϕ) + 0.0*(φ⋅dv) + ((dF∘(∇(u),∇(du)))⋅(S∘∇(u)) + (F∘∇(u))⋅(dS∘(∇(u),∇(du)))) ⊙ ∇(φ) )dΩs
end

# ### Temporal Jacobian
function jac_t(t,(u,v,p),(ut,vt,pt),(dut,dvt,dpt),(ϕ,φ,q))
	∫( ( (J∘∇(u)) * ρ_f * dvt ) ⋅ φ + 
	 - ( (J∘∇(u)) * ρ_f * (conv∘((Finv∘∇(u))⋅dut, ∇(v)) )) ⋅ φ )dΩf +
	∫( dut⋅ϕ + ρ_s*(dvt⋅φ) )dΩs
end

# ### Transient FE operator
op = TransientFEOperator(res,jac,jac_t,X,Y)

# ## Solve
# ### Stokes Solution
xh_ST = solve(op_ST)

# ### Nonlinear solver
nls = NLSolver(
	show_trace = true,
	method = :newton,
	xtol = 1e-7,
	ftol = 1e-7,
	linesearch = BackTracking(),
)

# ### Transient solver
Δt = 0.01
θ = 0.5
t₀ = 0.0
tf = 0.02
odes =  ThetaMethod(nls, Δt, θ)
solver = TransientFESolver(odes)

# ### Initial solution from Stokes
xh₀  = interpolate_everywhere(X(0.0),xh_ST)

# ### FSI solution
xhₜ = solve(solver, op, xh₀, t₀, tf)

# ## Output
filePath = "FSI2"
outfiles = paraview_collection(filePath, append=true) do pvd
	for (i, (xh, t)) in enumerate(xhₜ)
		pvd[t] = createvtk(Ω,filePath * "_$t.vtu",cellfields = ["uh" => uh, "vh" => vh, "ph" => ph])
	end
end

end
