using Gridap
using GridapODEs
using LinearAlgebra: tr, inv, det
import GridapODEs.TransientFETools: ∂t

model = DiscreteModelFromFile("../runs/elasticFlag_coarse.json")
model_solid = DiscreteModel(model,"solid")
model_fluid = DiscreteModel(model,"fluid")
writevtk(model,"model")

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
const (λ_s,μ_s) = lame_parameters(1.0, 0.33)
const (λ_m,μ_m) = lame_parameters(1.0, 0.33)
const μ_f = Um * ⌀ / Re
const ρ_s = 1.0
const ρ_f = 1.0

# ### Constitutive laws
@law F(∇u) = ∇u + one(∇u)
@law J(∇u) = det(F(∇u))
@law Finv(∇u) = inv(F(∇u))
@law FinvT(∇u) = (Finv(∇u)')
@law E(∇u) = 0.5 * ((F(∇u)')*F(∇u) - one(F(∇u)))
@law S(∇u) = 2*μ_s*E(∇u) + λ_s*tr(E(∇u))*one(E(∇u))
@law σ_dev(∇v,Finv) = μ_f*(∇v*Finv + (Finv')*(∇v'))
@law σ_m(ε) = λ_m*tr(ε)*one(ε) + 2*μ_m*ε
@law conv(c,∇v) = (∇v') * c
@law dconv(dc,∇dv,c,∇v) = conv(c,∇dv) + conv(dc,∇v)

# Derivatives:
@law dF(∇du) = ∇du
@law dJ(∇u,∇du) = J(F(∇u))*tr(inv(F(∇u))*dF(∇du))
@law dE(∇u,∇du) = 0.5 * ((dF(∇du)')*F(∇u) + (F(∇u)')*dF(∇du))
@law dS(∇u,∇du) = 2*μ_s*dE(∇u,∇du) + λ_s*tr(dE(∇u,∇du))*one(E(∇u))
@law dFinv(∇u,∇du) = -Finv(∇u) * dF(∇du) * Finv(∇u)
@law dFinvT(∇u,∇du) = (dFinv(∇u,∇du)')


k = 2
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
    valuetype=VectorValue{2,Float64},
    reffe=:Lagrangian,
    order=k,
    conformity =:H1,
    dirichlet_tags=["inlet", "noSlip", "cylinder","interface","outlet"])
Vvf = TestFESpace(
    model=model_fluid,
    valuetype=VectorValue{2,Float64},
    reffe=:Lagrangian,
    order=k,
    conformity =:H1,
    dirichlet_tags=["inlet", "noSlip", "cylinder","interface"])
Qf = TestFESpace(
		model=model_fluid,
		valuetype=Float64,
		order=k-1,
		reffe=:Lagrangian,
		conformity=:C0)

Uuf = TrialFESpace(Vuf,[u_noSlip(0.0), u_noSlip(0.0), u_noSlip(0.0), u_noSlip(0.0), u_noSlip(0.0)])
Uvf = TrialFESpace(Vvf,[u_in(0.0), u_noSlip(0.0), u_noSlip(0.0), u_noSlip(0.0)])
Utu = TransientTrialFESpace(Vu,[u_noSlip, u_noSlip, u_noSlip, u_noSlip, u_noSlip])
Utv = TransientTrialFESpace(Vv,[u_in, u_noSlip, u_noSlip, u_noSlip])
Pf = TrialFESpace(Qf)

Y = MultiFieldFESpace([Vuf,Vvf,Qf])
X = MultiFieldFESpace([Uuf,Uvf,Pf])
Yt = MultiFieldFESpace([Vu,Vv,Qf])
Xt = MultiFieldFESpace([Ut_u,Ut_v,Pf])

function a_vp_f(x, y)
    v, p = x
    φ, q = y
		inner(ε(φ), 2*μ_f*ε(v)) - (∇ * φ) * p + q * (∇ * v)
end

function a_uvp_f(x, y)
    u, v, p = x
    ϕ, φ, q = y
		inner(∇(ϕ), σ_m(ε(u))) + inner(ε(φ), 2*μ_f*ε(v)) - (∇ * φ) * p + q * (∇ * v)
end

function at_uvp_f(x, xt, y)
    u, v, p = x
		ut, vt, pt = xt
    ϕ, φ, q = y
		inner(φ,vt)
end

function a_uv_s(x, y)
    u,v = x
    ϕ,φ = y
		inner(ε(ϕ), σ_m(ε(u))) + inner(ε(φ), 2*μ_f*ε(v))
end

function at_uv_s(x, xt, y)
    u,v = x
		ut, vt = xt
    ϕ,φ = y
		inner(ϕ,ut) + inner(φ,vt)
end

res_uvp_f(x,y) = a_uvp_f(x,y)
res_uvp_f(t,x,xt,y) =	at_uvp_f(x,xt,y) + a_uvp_f(x,y)
res_uv_s(t,x,xt,y) = at_uvp_s(x,xt,y) + a_uvp_s(x,y)

jac_uvp_f(x,dx,y) = a_uvp_f(dx,y)
jac_uvp_f(t,x,xt,dx,y) = a_uvp_f(dx,y)
jac_t_uvp_f(t,x,xt,dxt,y) =	at_uvp_f(x,dxt,y)
jac_uv_s(t,x,xt,dx,y) =	a_uv_s(dx,y)
jac_t_uv_s(t,x,xt,dxt,y) = at_uv_s(x,dxt,y)

trian = Triangulation(model)
trian_solid = Triangulation(model_solid)
trian_fluid = Triangulation(model_fluid)
degree = 2*k
quad_solid = CellQuadrature(trian_solid,degree)
quad_fluid = CellQuadrature(trian_fluid,degree)

t_Ωf = FETerm(res_uvp_f, jac_uvp_f, trian_fluid, quad_fluid)
op = FEOperator(X,Y,t_Ωf)
xh = solve(op)

# vh = restrict(xh.blocks[1],trian_fluid)
# ph = restrict(xh.blocks[2],trian_fluid)
# writevtk(trian_fluid,"test_vp_f_0.0.vtu",cellfields = ["vh" => vh, "ph" => ph])

# U0u = Utu(0.0)
# uh0 = interpolate_everywhere(U0u, u_noSlip(0.0))
# vh0 = xh[1]
# ph0 = xh[2]
# xh0 = Gridap.MultiField.MultiFieldFEFunction(Xt(0.0),[uh0,vh0,ph0])
xh0  = interpolate(Xt(0.0),xh)
uh0f = restrict(xh0.blocks[1],trian_fluid)
vh0f = restrict(xh0.blocks[2],trian_fluid)
ph0f = restrict(xh0.blocks[3],trian_fluid)
writevtk(trian_fluid,"test_uvp_f_0.0.vtu",cellfields = ["uh" => uh0f, "vh" => vh0f, "ph" => ph0f])
writevtk(trian,"test_uvp_0.0.vtu",cellfields = ["uh" => xh0[1], "vh" => xh0[2], "ph" => xh0[3]])

t_t_Ωf = FETerm(res_uvp_f, jac_uvp_f, jac_t_uvp_f, trian_fluid, quad_fluid)
t_t_Ωs = FETerm(res_uv_s, jac_uv_s, jac_t_uv_s, trian_solid, quad_solid)
opt = TransientFEOperator(Xt,Yt,t_t_Ωf,t_t_Ωs)
ls = LUSolver()
odes =  ThetaMethod(ls, 0.001, 0.5)
solver = TransientFESolver(odes)
sol_t = solve(solver, opt, xh0, 0.0, 0.002)
println(length(xh0))
for (xh,t) in sol_t
		println(typeof(t), length(xh[1]))
		# uh = restrict(xh.blocks[1],trian_fluid)
		# vh = restrict(xh.blocks[2],trian_fluid)
		# ph = restrict(xh.blocks[3],trian_fluid)
		# writevtk(trian_fluid,"test_uvp_f_$t.vtu",cellfields = ["uh" => uh, "vh" => vh, "ph" => ph])
		writevtk(trian,"test_uvp_$t.vtu",cellfields = ["uh" => xh0[1], "vh" => xh0[2], "ph" => xh0[3]])
end


# vh = restrict(xh.blocks[1],trian_fluid)
# ph = restrict(xh.blocks[2],trian_fluid)
# writevtk(trian_fluid,"test.vtu",cellfields = ["vh" => vh, "ph" => ph])
