using Gridap
using GridapODEs.ODETools
using GridapODEs.TransientFETools
using LinearAlgebra: tr, inv, det
using LineSearches: BackTracking
using WriteVTK
import GridapODEs.TransientFETools: ∂t

#model = DiscreteModelFromFile("../runs/elasticFlag_coarse.json")
model = DiscreteModelFromFile("../models/elasticFlag.json")
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
const (λ_m,μ_m) = lame_parameters(1.0e-6, 0.33)
const (λ_s,μ_s) = lame_parameters(1.4e6, 0.4)
const ρ_s = 1.0e4
const ρ_f = 1.0e3
const μ_f = ρ_f * Um * ⌀ / Re

# ### Constitutive laws
@law F(∇u) = ∇u + one(∇u)
@law J(∇u) = det(F(∇u))
@law Finv(∇u) = inv(F(∇u))
@law FinvT(∇u) = (Finv(∇u)')
@law E(∇u) = 0.5 * ((F(∇u)')⋅F(∇u) - one(F(∇u)))
@law S(∇u) = 2*μ_s*E(∇u) + λ_s*tr(E(∇u))*one(E(∇u))
@law σ_dev(∇v,Finv) = μ_f*(∇v⋅Finv + (Finv')⋅(∇v'))
@law σ_m(ε) = λ_m*tr(ε)*one(ε) + 2*μ_m*ε
@law conv(c,∇v) = (∇v') ⋅ c
@law Sm(∇u) = 2*μ_m*E(∇u) + λ_m*tr(E(∇u))*one(E(∇u))

# Derivatives:
@law dF(∇du) = ∇du
@law dJ(∇u,∇du) = J(F(∇u))*tr(inv(F(∇u))⋅dF(∇du))
@law dE(∇u,∇du) = 0.5 * ((dF(∇du)')⋅F(∇u) + (F(∇u)')⋅dF(∇du))
@law dS(∇u,∇du) = 2*μ_s*dE(∇u,∇du) + λ_s*tr(dE(∇u,∇du))*one(E(∇u))
@law dFinv(∇u,∇du) = -Finv(∇u) ⋅ dF(∇du) ⋅ Finv(∇u)
@law dFinvT(∇u,∇du) = (dFinv(∇u,∇du)')
@law dconv(dc,∇dv,c,∇v) = conv(c,∇dv) + conv(dc,∇v)
@law dSm(∇u,∇du) = 2*μ_m*dE(∇u,∇du) + λ_m*tr(dE(∇u,∇du))*one(E(∇u))

# Test FE Spaces
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
		(∇(ϕ) ⊙ σ_m(ε(u)))
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
function aFSI_ϕ_f(x, y)
    u, v, p = x
    ϕ, φ, q = y
		(∇(ϕ) ⊙ (F(∇(u))⋅Sm(∇(u))))
		#(∇(ϕ) ⊙ σ_m(ε(u)))
end
function aFSI_ϕ_Γi(x,y)
		u,v,p = x
    ϕ,φ,q = y
		- (ϕ ⋅  (n_Γi⋅(F(∇(u))⋅Sm(∇(u)))) )
    #- (ϕ ⋅  (n_Γi⋅σ_m(ε(u))))
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
		q * (∇⋅v)
		#q * (∇⋅( (J(∇(u))*Finv(∇(u)))⋅v ))

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
function daFSI_du_ϕ_f(x,dx, y)
		u, v, p = x
    du, dv, dp = dx
    ϕ, φ, q = y
		(∇(ϕ) ⊙ (dF(∇(du))⋅Sm(∇(u)) + F(∇(u))⋅dSm(∇(u),∇(du))))
		#(∇(ϕ) ⊙ σ_m(ε(du)))
end
function daFSI_du_ϕ_Γi(x,dx,y)
		u,v,p = x
		du,dv,dp = dx
    ϕ,φ,q = y
    - (ϕ ⋅  (n_Γi⋅(dF(∇(du))⋅Sm(∇(u)) + F(∇(u))⋅dSm(∇(u),∇(du)))) )
    #- (ϕ ⋅  (n_Γi⋅σ_m(ε(du))) )
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

# Triangulations and quadratures
degree = 2*k
bdegree = 2*k
trian = Triangulation(model)
trian_solid = Triangulation(model_solid)
trian_fluid = Triangulation(model_fluid)
trian_Γi = BoundaryTriangulation(model_fluid,"interface")
n_Γi = get_normal_vector(trian_Γi)
quad_solid = CellQuadrature(trian_solid,degree)
quad_fluid = CellQuadrature(trian_fluid,degree)
quad_Γi = CellQuadrature(trian_Γi,bdegree)

# Stokes FE Operator
res_ST_f(x,y) = aST_ϕ_f(x,y) + aST_φ_f(x,y) + aST_q_f(x,y)
jac_ST_f(x,dx,y) = aST_ϕ_f(dx,y) + aST_φ_f(dx,y) + aST_q_f(dx,y)
tST_Ωf = FETerm(res_ST_f, jac_ST_f, trian_fluid, quad_fluid)
opST = FEOperator(X,Y,tST_Ωf)

# FSI FE Operator
res_FSI_f(t,x,xt,y) =	aFSI_ϕ_f(x,y) + aFSI_φ_f(x,xt,y) + aFSI_q_f(x,y)
jac_FSI_f(t,x,xt,dx,y) = 	daFSI_du_ϕ_f(x,dx,y) + daFSI_du_φ_f(x,xt,dx,y) + daFSI_dv_φ_f(x,xt,dx,y) + daFSI_dp_φ_f(x,dx,y) + aFSI_q_f(dx,y)
jac_t_FSI_f(t,x,xt,dxt,y) =	 daFSI_dut_φ_f(x,dxt,y) + daFSI_dvt_φ_f(x,dxt,y)
res_FSI_s(t,x,xt,y) = aFSI_ϕ_s(x,xt,y) + aFSI_φ_s(x,xt,y)
jac_FSI_s(t,x,xt,dx,y) = daFSI_ϕ_s(x,dx,y) + daFSI_φ_s(x,dx,y)
jac_t_FSI_s(t,x,xt,dxt,y) = daFSI_dt_s(x,dxt,y)
tFSI_Ωf = FETerm(res_FSI_f, jac_FSI_f, jac_t_FSI_f, trian_fluid, quad_fluid)
tFSI_Ωs = FETerm(res_FSI_s, jac_FSI_s, jac_t_FSI_s, trian_solid, quad_solid)
tFSI_Γi = FETerm(aFSI_ϕ_Γi,daFSI_du_ϕ_Γi,trian_Γi,quad_Γi)
opFSI = TransientFEOperator(Xt,Yt,tFSI_Ωf,tFSI_Ωs,tFSI_Γi)

# Output function
function writePVD(filePath::String, trian::Triangulation, sol; append=false)
    outfiles = paraview_collection(filePath, append=append) do pvd
        for (i, (xh, t)) in enumerate(sol)
            uh = xh.blocks[1]
            vh = xh.blocks[2]
            ph = xh.blocks[3]
            pvd[t] = createvtk(
                trian,
                filePath * "_$t.vtu",
                cellfields = ["uh" => uh, "vh" => vh, "ph" => ph],
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
writePVD(filePath, trian, [(xh, 0.0)])

# Solve FSI problem
xh0  = interpolate(Xt(0.0),xh)
nls = NLSolver(
    show_trace = true,
    method = :newton,
    linesearch = BackTracking(),
)
odes =  ThetaMethod(nls, 0.05, 0.5)
solver = TransientFESolver(odes)
sol_t = solve(solver, opFSI, xh0, 0.0, 10.0)
writePVD(filePath, trian, sol_t, append=true)


# println(length(xh0.blocks))
# for (xh,t) in sol_t
# 		println(typeof(t), length(xh[1]))
# 		# uh = restrict(xh.blocks[1],trian_fluid)
# 		# vh = restrict(xh.blocks[2],trian_fluid)
# 		# ph = restrict(xh.blocks[3],trian_fluid)
# 		# writevtk(trian_fluid,"test_uvp_f_$t.vtu",cellfields = ["uh" => uh, "vh" => vh, "ph" => ph])
# 		writevtk(trian,"test_uvp_$t.vtu",cellfields = ["uh" => xh0[1], "vh" => xh0[2], "ph" => xh0[3]])
# end

# vh = restrict(xh.blocks[1],trian_fluid)
# ph = restrict(xh.blocks[2],trian_fluid)
# writevtk(trian_fluid,"test.vtu",cellfields = ["vh" => vh, "ph" => ph])

