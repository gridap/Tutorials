using Gridap
using Gridap.Arrays
using Gridap.Geometry
using GridapODEs.ODETools
using GridapODEs.TransientFETools
using LinearAlgebra: tr, inv, det
using LineSearches: BackTracking, HagerZhang, InitialHagerZhang
using WriteVTK
using TimerOutputs
import GridapODEs.TransientFETools: ∂t


######## SOLVER #######
using Gridap.Algebra
using IterativeSolvers
using IncompleteLU
using Preconditioners: AMGPreconditioner, SmoothedAggregation
import Gridap.Algebra: LinearSolver
import Gridap.Algebra: symbolic_setup
import Gridap.Algebra: numerical_setup
import Gridap.Algebra: numerical_setup!
import Gridap.Algebra: solve!

reset_timer!()

struct GmresSolver <: LinearSolver
  preconditioner
  precond_kwargs::Dict
  function GmresSolver(;
    preconditioner=IterativeSolvers.Identity, precond_kwargs...)
    new(preconditioner,precond_kwargs)
  end
end

struct GmresSymbolicSetup <: SymbolicSetup
  preconditioner
  precond_kwargs::Dict
end

mutable struct GmresNumericalSetup{T<:AbstractMatrix} <: NumericalSetup
  A::T
  preconditioner
  precond_kwargs::Dict
end

function symbolic_setup(s::GmresSolver, mat::AbstractMatrix)
  GmresSymbolicSetup(s.preconditioner, s.precond_kwargs)
end

function numerical_setup(ss::GmresSymbolicSetup,mat::AbstractMatrix)
  GmresNumericalSetup(mat, ss.preconditioner, ss.precond_kwargs)
end

function numerical_setup!(ns::GmresNumericalSetup, mat::AbstractMatrix)
  ns.A = mat
end

function get_preconditioner(ns::GmresNumericalSetup)

end

function solve!(
  x::AbstractVector,ns::GmresNumericalSetup,b::AbstractVector)
  p=ns.preconditioner(ns.A; ns.precond_kwargs...)
  # initialize the solution to 0
  x .= 0.0
  gmres!(x, ns.A, b,verbose=false, Pl=p, initially_zero=true, tol=1e-4)
end
#######################


#model = DiscreteModelFromFile("../runs/elasticFlag_coarse.json")
#model = DiscreteModelFromFile("../models/elasticFlagFine.json")
model = DiscreteModelFromFile("../models/elasticFlag.json")
model_solid = DiscreteModel(model,"solid")
model_fluid = DiscreteModel(model,"fluid")
writevtk(model,"model")

# Triangulations and quadratures
trian = Triangulation(model)
trian_solid = Triangulation(model_solid)
trian_fluid = Triangulation(model_fluid)
trian_Γi = BoundaryTriangulation(model_fluid,"interface")
k = 2
degree = 2*k
bdegree = 2*k
quad_solid = CellQuadrature(trian_solid,degree)
quad_fluid = CellQuadrature(trian_fluid,degree)
quad_Γi = CellQuadrature(trian_Γi,bdegree)
n_Γi = get_normal_vector(trian_Γi)

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

# Compute cell area (auxiliar quantity for mesh motion eq.)
xe = get_cell_coordinates(trian_fluid)
a(x)::Float64 = abs(x[1][1]*(x[2][2]-x[3][2]) + x[2][1]*(x[3][2]-x[1][2]) + x[3][1]*(x[1][2]-x[2][2])) / 2
ae = apply(Float64, x->a(x),xe)
#α = apply(Float64, a->1/a, ae)
vol = cell_measure(trian_fluid,trian)
vol_fluid = reindex(vol,trian_fluid)#integrate(1,trian_fluid,CellQuadrature(trian_fluid,0))#cell_measure(trian_fluid,trian)
vol_Γi = reindex(vol,trian_Γi)

# ### Material properties
function lame_parameters(E,ν)
		λ = (E*ν)/((1+ν)*(1-2*ν))
		μ = E/(2*(1+ν))
		(λ, μ)
end
const (λ_s,μ_s) = lame_parameters(1.4e6, 0.4)
const ρ_s = 1.0e4
const ρ_f = 1.0e3
const μ_f = ρ_f * Um * ⌀ / Re
const β_m = 4 # β_m = λ_m / μ_m
const E_m = 1.0
const ν_m = 0.25

# ### Constitutive laws
@law F(∇u) = ∇u + one(∇u)
@law J(∇u) = det(F(∇u))
@law Finv(∇u) = inv(F(∇u))
@law FinvT(∇u) = (Finv(∇u)')
@law E(∇u) = 0.5 * ((F(∇u)')⋅F(∇u) - one(F(∇u)))
@law S(∇u) = 2*μ_s*E(∇u) + λ_s*tr(E(∇u))*one(E(∇u))
@law σ_dev(∇v,Finv) = μ_f*(∇v⋅Finv + (Finv')⋅(∇v'))
@law σ_m(α,ε) = λ_m(α)*tr(ε)*one(ε) + 2.0*μ_m(α)*ε
#@law σ_m(∇ut,Finv) = λ_m*tr(∇ut⋅Finv)*one(Finv) + μ_m*(∇ut⋅Finv + (Finv')⋅(∇ut')) 
#@law σ_m(∇ut,Finv) = β_m*tr(∇ut⋅Finv)*one(Finv) + (∇ut⋅Finv + (Finv')⋅(∇ut')) 
#@law σ_m2(α,∇ut,Finv) = 16.0*α*μ_s*tr(∇ut⋅Finv)*one(Finv) + α*μ_s*(∇ut⋅Finv + (Finv')⋅(∇ut')) 
@law conv(c,∇v) = (∇v') ⋅ c
@law Sm(∇u) = 2*μ_m*E(∇u) + λ_m*tr(E(∇u))*one(E(∇u))
@law α(ve,J) = 1.0e-5/(J)
@law λ_m(α) = α * (E_m*ν_m) / ((1.0+ν_m)*(1.0-2.0*ν_m))
@law μ_m(α) = α * E_m / (2.0*(1.0+ν_m))
#@law λ_m(J) = (E_m*ν_m) / ((1.0+ν_m)*(1.0-2.0*ν_m))
#@law μ_m(J) = E_m / (2.0*(1.0+ν_m))

# Derivatives:
@law dF(∇du) = ∇du
@law dJ(∇u,∇du) = J(F(∇u))*tr(inv(F(∇u))⋅dF(∇du))
@law dE(∇u,∇du) = 0.5 * ((dF(∇du)')⋅F(∇u) + (F(∇u)')⋅dF(∇du))
@law dS(∇u,∇du) = 2*μ_s*dE(∇u,∇du) + λ_s*tr(dE(∇u,∇du))*one(E(∇u))
@law dFinv(∇u,∇du) = -Finv(∇u) ⋅ dF(∇du) ⋅ Finv(∇u)
@law dFinvT(∇u,∇du) = (dFinv(∇u,∇du)')
@law dconv(dc,∇dv,c,∇v) = conv(c,∇dv) + conv(dc,∇v)
@law dSm(∇u,∇du) = 2*μ_m*dE(∇u,∇du) + λ_m*tr(dE(∇u,∇du))*one(E(∇u))
@law dα(ve,J,dJ) = - 1.0e-5 * 1.0 / (J*J) * dJ
@law dλ_m(dα) = dα * (E_m*ν_m) / ((1.0+ν_m)*(1.0-2.0*ν_m))
@law dμ_m(dα) = dα * E_m / (2.0*(1.0+ν_m))
@law dσ_m(dα,ε) = dλ_m(dα)*tr(ε)*one(ε) + 2.0*dμ_m(dα)*ε

# Test FE Spaces
Vu = TestFESpace(
    model=model,
    valuetype=VectorValue{2,Float64},
    reffe=:Lagrangian,
    order=k,
    conformity =:H1,
    dirichlet_tags=["inlet", "noSlip", "cylinder","fixed","outlet"])
Vv = TestFESpace(
    model=model,
    valuetype=VectorValue{2,Float64},
    reffe=:Lagrangian,
    order=k,
    conformity =:H1,
    dirichlet_tags=["inlet", "noSlip", "cylinder","fixed"])
Vuf = TestFESpace(
    model=model_fluid,
		#model=model,
    valuetype=VectorValue{2,Float64},
    reffe=:Lagrangian,
    order=k,
    conformity =:H1,
    dirichlet_tags=["inlet", "noSlip", "cylinder","interface","outlet"])
Vvf = TestFESpace(
    model=model_fluid,
		#model=model,
    valuetype=VectorValue{2,Float64},
    reffe=:Lagrangian,
    order=k,
    conformity =:H1,
    dirichlet_tags=["inlet", "noSlip", "cylinder","interface"])
Qf = TestFESpace(
		model=model_fluid,
		#model=model,
		valuetype=Float64,
		order=k-1,
		reffe=:Lagrangian,
		conformity=:C0)

# Trial FE Spaces
Uuf = TrialFESpace(Vuf,[u_noSlip(0.0), u_noSlip(0.0), u_noSlip(0.0), u_noSlip(0.0), u_noSlip(0.0)])
Uvf = TrialFESpace(Vvf,[u_in(0.0), u_noSlip(0.0), u_noSlip(0.0), u_noSlip(0.0)])
Utu = TransientTrialFESpace(Vu,[u_noSlip, u_noSlip, u_noSlip, u_noSlip, u_noSlip])
Utv = TransientTrialFESpace(Vv,[u_in, u_noSlip, u_noSlip, u_noSlip])
Pf = TrialFESpace(Qf)

# Multifield FE Spaces
Y = MultiFieldFESpace([Vuf,Vvf,Qf])
X = MultiFieldFESpace([Uuf,Uvf,Pf])
Yt = MultiFieldFESpace([Vu,Vv,Qf])
Xt = MultiFieldFESpace([Utu,Utv,Pf])

# Stokes Bilinear forms
function aST_ϕ_f(x, y)
    u, v, p = x
    ϕ, φ, q = y
		(∇(ϕ) ⊙ ∇(u))
end
function aST_φ_f(x, y)
    u, v, p = x
    ϕ, φ, q = y
		visc_term = ε(φ) ⊙ ( 2*μ_f*ε(v) )
		pres_term = (∇⋅φ) * p

		visc_term + pres_term
end
function aST_q_f(x, y)
    u, v, p = x
    ϕ, φ, q = y
		q * (∇⋅v)
end

# FSI Bilinear forms
function aFSI_ϕ_f(x, xt, y)
    u, v, p = x
    ut, vt, pt = xt
    ϕ, φ, q = y
		#(∇(ϕ) ⊙ (F(∇(u))⋅Sm(∇(u))))
		#(∇(ϕ) ⊙ (J(∇(u)) * (σ_m(aJ(J(∇(u))),∇(ut),Finv(∇(u))) ⋅ FinvT(∇(u)))) )
		#println(typeof(aJ),typeof(∇(ut)),typeof(Finv(∇(u))))
		#α(vol,J(∇(u))) * ( ∇(ϕ) ⊙ (J(∇(u)) * (σ_m(∇(ut),Finv(∇(u))) ⋅ FinvT(∇(u)))) )
		#α(vol_fluid,J(∇(u))) * (∇(ϕ) ⊙ σ_m(ε(ut))) 
		#J2inv(J(∇(u))) * (∇(ϕ) ⊙ σ_m(ε(ut))) 
		#aJ = ae*(J(∇(u)))
		#println(aJ(J(∇(u))))
		(∇(ϕ) ⊙ σ_m(α(vol_fluid,J(∇(u))),ε(ut)))
end
function aFSI_ϕ_Γi(x,xt,y)
		u,v,p = x
    ut, vt, pt = xt
    ϕ,φ,q = y
		#- (ϕ ⋅  (n_Γi⋅(F(∇(u))⋅Sm(∇(u)))) )
    #- (ϕ ⋅  (n_Γi⋅σ_m(ε(u))))
		#- α(bvol,J(∇(u))) * (ϕ ⋅ (n_Γi⋅(J(∇(u)) * (σ_m(∇(ut),Finv(∇(u))) ⋅ FinvT(∇(u)))) ))
    #- α(vol_Γi,J(∇(u))) * (ϕ ⋅  (n_Γi⋅σ_m(ε(ut))))
    #- J2inv(J(∇(u))) * (ϕ ⋅  (n_Γi⋅σ_m(ε(ut))))
    - (ϕ ⋅  (n_Γi⋅σ_m(α(vol_Γi,J(∇(u))),ε(ut))) )
end
function aFSI_φ_f(x, xt, y)
    u, v, p = x
    ut, vt, pt = xt
    ϕ, φ, q = y
		temp_term = φ ⋅ ( J(∇(u)) * ρ_f * vt )
		conv_term = φ ⋅ ( J(∇(u)) * ρ_f * conv(	Finv(∇(u))⋅(v-ut), ∇(v)) )
		visc_term = ( ∇(φ) ⊙ ( J(∇(u)) * σ_dev(∇(v),Finv(∇(u))) ⋅ FinvT(∇(u))) )
		pres_term = (∇⋅φ) * J(∇(u)) * p * tr(FinvT(∇(u)))
		temp_term + conv_term + visc_term + pres_term
end
function aFSI_q_f(x, y)
    u, v, p = x
    ϕ, φ, q = y
		#q * (∇⋅v)
		q * (J(∇(u))*(∇(v)⊙FinvT(∇(u))))

end
function aFSI_ϕ_s(x, xt, y)
    u,v,p = x
		ut, vt,pt = xt
    ϕ,φ,q = y
		(ϕ⋅ut) + 0.0*(u⋅ϕ) - (ϕ⋅v) 
end
function aFSI_φ_s(x, xt, y)
    u,v,p = x
		ut, vt,pt = xt
    ϕ,φ,q = y
		(φ⋅(ρ_s*vt)) + 0.0*(φ⋅(ρ_s*v)) + (∇(φ) ⊙ (F(∇(u))⋅S(∇(u))))
end

# FSI Jacobians
function daFSI_du_ϕ_f(x, xt, dx, y)
		u, v, p = x
		ut, vt, pt = xt
    du, dv, dp = dx
    ϕ, φ, q = y
		#(∇(ϕ) ⊙ (dF(∇(du))⋅Sm(∇(u)) + F(∇(u))⋅dSm(∇(u),∇(du))))
		#dα(vol_fluid,J(∇(u)),dJ(∇(u),∇(du))) * (∇(ϕ) ⊙ σ_m(ε(ut)))
		#dJ2inv(J(∇(u)),dJ(∇(u),∇(du))) * (∇(ϕ) ⊙ σ_m(ε(ut)))
		#dα(vol,J(∇(u)),dJ(∇(u),∇(du))) * ( ∇(ϕ) ⊙ (J(∇(u)) * (σ_m(∇(ut),Finv(∇(u))) ⋅ FinvT(∇(u)))) ) + α(vol,J(∇(u))) * (∇(ϕ) ⊙ (dJ(∇(u),∇(du)) * (σ_m(∇(ut),Finv(∇(u))) ⋅ FinvT(∇(u))) + J(∇(u)) * (σ_m(∇(ut),dFinv(∇(u),∇(du))) ⋅ FinvT(∇(u))) + J(∇(u)) * (σ_m(∇(ut),Finv(∇(u))) ⋅ dFinvT(∇(u),∇(du)))) )
		#(∇(ϕ) ⊙ σ_m(ε(du)))
		(∇(ϕ) ⊙ dσ_m(dα(vol_fluid,J(∇(u)),dJ(∇(u),∇(du))),ε(ut)))
end
function daFSI_dut_ϕ_f(x, dxt, y)
		u, v, p = x
    dut, dvt, dpt = dxt
    ϕ, φ, q = y
		#(∇(ϕ) ⊙ (dF(∇(du))⋅Sm(∇(u)) + F(∇(u))⋅dSm(∇(u),∇(du))))
		#(∇(ϕ) ⊙ σ_m(ε(du)))
		#α(vol,J(∇(u))) * (∇(ϕ) ⊙ (J(∇(u)) * (σ_m(∇(dut),Finv(∇(u))) ⋅ FinvT(∇(u))) ) )
		#α(vol_fluid,J(∇(u))) * (∇(ϕ) ⊙ σ_m(ε(dut))) 
		#J2inv(J(∇(u))) * (∇(ϕ) ⊙ σ_m(ε(dut)))
		(∇(ϕ) ⊙ σ_m(α(vol_fluid,J(∇(u))),ε(dut))) 
end
function daFSI_du_ϕ_Γi(x,xt,dx,y)
		u,v,p = x
    ut, vt, pt = xt
		du,dv,dp = dx
    ϕ,φ,q = y
    #- (ϕ ⋅  (n_Γi⋅(dF(∇(du))⋅Sm(∇(u)) + F(∇(u))⋅dSm(∇(u),∇(du)))) )
    #- dα(vol_Γi,J(∇(u)),dJ(∇(u),∇(du))) * (ϕ ⋅  (n_Γi⋅σ_m(ε(ut))) )
    #- dJ2inv(J(∇(u)),dJ(∇(u),∇(du))) * (ϕ ⋅  (n_Γi⋅σ_m(ε(ut))) )
		#- dα(bvol,J(∇(u)),dJ(∇(u),∇(du))) * (ϕ ⋅ (n_Γi⋅(J(∇(u)) * (σ_m(∇(ut),Finv(∇(u))) ⋅ FinvT(∇(u)))) )) - α(bvol,J(∇(u))) * (ϕ ⋅ (n_Γi⋅(dJ(∇(u),∇(du)) * (σ_m(∇(ut),Finv(∇(u))) ⋅ FinvT(∇(u))) + J(∇(u)) * (σ_m(∇(ut),dFinv(∇(u),∇(du))) ⋅ FinvT(∇(u))) + J(∇(u)) * (σ_m(∇(ut),Finv(∇(u))) ⋅ dFinvT(∇(u),∇(du)))) ))
    - (ϕ ⋅  (n_Γi⋅dσ_m(dα(vol_Γi,J(∇(u)),dJ(∇(u),∇(du))),ε(ut))) )
end
function daFSI_dut_ϕ_Γi(x,dxt,y)
		u,v,p = x
		dut,dvt,dpt = dxt
    ϕ,φ,q = y
    #- (ϕ ⋅  (n_Γi⋅(dF(∇(du))⋅Sm(∇(u)) + F(∇(u))⋅dSm(∇(u),∇(du)))) )
    #- (ϕ ⋅  (n_Γi⋅σ_m(ε(du))) )
		#- α(bvol,J(∇(u))) * (ϕ ⋅ (n_Γi⋅(J(∇(u)) * (σ_m(∇(dut),Finv(∇(u))) ⋅ FinvT(∇(u)))) ))
		#- α(vol_Γi,J(∇(u))) * (ϕ ⋅  (n_Γi⋅σ_m(ε(dut))) )
		#- J2inv(J(∇(u))) * (ϕ ⋅  (n_Γi⋅σ_m(ε(dut))) )
		- (ϕ ⋅  (n_Γi⋅σ_m(α(vol_Γi,J(∇(u))),ε(dut))) ) 
end
function daFSI_du_φ_f(x, xt, dx, y)
    u, v, p = x
    ut, vt, pt = xt
    du, dv, dp = dx
    ϕ, φ, q = y
		temp_term = φ ⋅ ( dJ(∇(u),∇(du)) * ρ_f * vt )
		conv_term = φ ⋅ ( ( dJ(∇(u),∇(du)) * ρ_f * conv( Finv(∇(u))⋅(v-ut), ∇(v)) ) +
											( J(∇(u)) * ρ_f * conv(	dFinv(∇(u),∇(du))⋅(v-ut), ∇(v)) ) )
		visc_term = ∇(φ) ⊙ ( dJ(∇(u),∇(du)) * σ_dev(∇(v),Finv(∇(u))) ⋅ FinvT(∇(u)) +
												 J(∇(u)) * σ_dev(∇(v),dFinv(∇(u),∇(du))) ⋅ FinvT(∇(u)) +
												 J(∇(u)) * σ_dev(∇(v),Finv(∇(u))) ⋅ dFinvT(∇(u),∇(du)) )
		pres_term = (∇⋅φ) * p * ( dJ(∇(u),∇(du)) * tr(FinvT(∇(u))) +
															J(∇(u)) * tr(dFinvT(∇(u),∇(du))) )
		temp_term + conv_term + visc_term + pres_term
end
function daFSI_dv_φ_f(x, xt, dx, y)
    u, v, p = x
    ut, vt, pt = xt
    du, dv, dp = dx
    ϕ, φ, q = y
		conv_term = φ ⋅ ( J(∇(u)) * ρ_f * dconv( Finv(∇(u))⋅dv, ∇(dv), Finv(∇(u))⋅(v-ut) , ∇(v)) )
		visc_term = ( ∇(φ) ⊙ ( J(∇(u)) * σ_dev(∇(dv),Finv(∇(u))) ⋅ FinvT(∇(u))) )
		conv_term + visc_term
end
function daFSI_dp_φ_f(x, dx, y)
    u, v, p = x
    du, dv, dp = dx
    ϕ, φ, q = y
		pres_term = (∇⋅φ) * J(∇(u)) * dp * tr(FinvT(∇(u)))
end
function daFSI_dut_φ_f(x, dxt, y)
    u, v, p = x
    dut, dvt, dpt = dxt
    ϕ, φ, q = y
		conv_term = - φ ⋅ ( J(∇(u)) * ρ_f * conv(	Finv(∇(u))⋅dut, ∇(v)) )
end
function daFSI_dvt_φ_f(x, dxt, y)
    u, v, p = x
    dut, dvt, dpt = dxt
    ϕ, φ, q = y
		temp_term = φ ⋅ ( J(∇(u)) * ρ_f * dvt )
end
function daFSI_du_q_f(x, dx, y)
    u, v, p = x
    du, dv, dp = dx
    ϕ, φ, q = y
		q * ( dJ(∇(u),∇(du))*(∇(v)⊙FinvT(∇(u))) + J(∇(u))*(∇(v)⊙dFinvT(∇(u),∇(du))) )
end
function daFSI_dv_q_f(x, dx, y)
    u, v, p = x
    du, dv, dp = dx
    ϕ, φ, q = y
		q * ( J(∇(u))*(∇(dv)⊙FinvT(∇(u))) )
end
function daFSI_ϕ_s(x, dx, y)
    u,v,p = x
    du,dv,dp = dx
    ϕ,φ,q = y
		0.0*(du⋅ϕ) - (ϕ⋅dv) 
end
function daFSI_φ_s(x, dx, y)
    u,v,p = x
    du,dv,dp = dx
    ϕ,φ,q = y
		0.0*(φ⋅(ρ_s*dv)) + (∇(φ) ⊙ ( dF(∇(du))⋅S(∇(u)) + (F(∇(u))⋅dS(∇(u),∇(du))) ) )
end
function daFSI_dt_s(x, dxt, y)
    u, v, p = x
    dut, dvt, dpt = dxt
    ϕ, φ, q = y
		ϕ⋅dut + (φ⋅(ρ_s*dvt))
end

# Stokes FE Operator
res_ST_f(x,y) = aST_ϕ_f(x,y) + aST_φ_f(x,y) + aST_q_f(x,y)
jac_ST_f(x,dx,y) = aST_ϕ_f(dx,y) + aST_φ_f(dx,y) + aST_q_f(dx,y)
tST_Ωf = FETerm(res_ST_f, jac_ST_f, trian_fluid, quad_fluid)
opST = FEOperator(X,Y,tST_Ωf)

# FSI FE Operator
res_FSI_f(t,x,xt,y) =	aFSI_ϕ_f(x,xt,y) + aFSI_φ_f(x,xt,y) + aFSI_q_f(x,y)
jac_FSI_f(t,x,xt,dx,y) = 	daFSI_du_ϕ_f(x,xt,dx,y) + daFSI_du_φ_f(x,xt,dx,y) + daFSI_dv_φ_f(x,xt,dx,y) + daFSI_dp_φ_f(x,dx,y) + daFSI_du_q_f(x,dx,y) + daFSI_dv_q_f(x,dx,y)
#jac_FSI_f(t,x,xt,dx,y) = 	daFSI_du_φ_f(x,xt,dx,y) + daFSI_dv_φ_f(x,xt,dx,y) + daFSI_dp_φ_f(x,dx,y) + daFSI_du_q_f(x,dx,y) + daFSI_dv_q_f(x,dx,y)
jac_t_FSI_f(t,x,xt,dxt,y) =	 daFSI_dut_ϕ_f(x,dxt,y) + daFSI_dut_φ_f(x,dxt,y) + daFSI_dvt_φ_f(x,dxt,y)
#jac_t_FSI_f(t,x,xt,dxt,y) =	 daFSI_dut_φ_f(x,dxt,y) + daFSI_dvt_φ_f(x,dxt,y)
res_FSI_s(t,x,xt,y) = aFSI_ϕ_s(x,xt,y) + aFSI_φ_s(x,xt,y)
jac_FSI_s(t,x,xt,dx,y) = daFSI_ϕ_s(x,dx,y) + daFSI_φ_s(x,dx,y)
jac_t_FSI_s(t,x,xt,dxt,y) = daFSI_dt_s(x,dxt,y)
res_FSI_fΓi(t,x,xt,y) = aFSI_ϕ_Γi(x,xt,y)
jac_FSI_fΓi(t,x,xt,dx,y) = 	daFSI_du_ϕ_Γi(x,xt,dx,y)
jac_t_FSI_fΓi(t,x,xt,dxt,y) = 	daFSI_dut_ϕ_Γi(x,dxt,y)
res_FSI_fΓi(x,y) = aFSI_ϕ_Γi(x,y)
jac_FSI_fΓi(x,dx,y) = 	daFSI_du_ϕ_Γi(x,dx,y)

tFSI_Ωf = FETerm(res_FSI_f, jac_FSI_f, jac_t_FSI_f, trian_fluid, quad_fluid)
tFSI_Ωs = FETerm(res_FSI_s, jac_FSI_s, jac_t_FSI_s, trian_solid, quad_solid)
tFSI_Γi = FETerm(res_FSI_fΓi,jac_FSI_fΓi,jac_t_FSI_fΓi,trian_Γi,quad_Γi)
#tFSI_Γi = FETerm(res_FSI_fΓi,jac_FSI_fΓi,trian_Γi,quad_Γi)
opFSI = TransientFEOperator(Xt,Yt,tFSI_Ωf,tFSI_Ωs,tFSI_Γi)

# Output function
function writePVD(filePath::String, trian::Triangulation, sol; append=false)
    outfiles = paraview_collection(filePath, append=append) do pvd
        for (i, (xh, t)) in enumerate(sol)
            uh = restrict(xh.blocks[1],trian)
            vh = restrict(xh.blocks[2],trian)
						ph = restrict(xh.blocks[3],trian)
						a = integrate(1.0/vol_fluid*J(∇(uh)),trian,CellQuadrature(trian,4))
            pvd[t] = createvtk(
                trian,
                filePath * "_$t.vtu",
                cellfields = ["uh" => uh, "vh" => vh, "ph" => ph],celldata = ["a"=>a]
            )
        end
    end
end
folderName = "fsi-results"
fileName = "fields"
if !isdir(folderName)
    mkdir(folderName)
end
filePath = join([folderName, fileName], "/")

# Solve Stokes problem
xh = solve(opST)
writePVD(filePath, trian_fluid, [(xh, 0.0)])

# Solve FSI problem
@timeit "FSI problem" begin
		xh0  = interpolate(Xt(0.0),xh)
		nls = NLSolver(
				#GmresSolver(preconditioner=ilu,τ=1.0e-6),
				#GmresSolver(preconditioner=AMGPreconditioner{SmoothedAggregation}),
				show_trace = true,
				method = :newton,
				#linesearch = HagerZhang(),
				linesearch = BackTracking(),
				ftol = 1.0e-6
		)
		odes =  ThetaMethod(nls, 0.1, 0.5)
		solver = TransientFESolver(odes)
		sol_t = solve(solver, opFSI, xh0, 0.0, 30.0)
		writePVD(filePath, trian_fluid, sol_t, append=true)
end
print_timer()
println()
