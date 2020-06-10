
#md # !!! note
#
#     This tutorial is under construction, but the code below is already functional.
#

using Gridap
using Gridap.Visualization
using GridapODEs.ODETools
using GridapODEs.TransientFETools
using Gridap.FESpaces: get_algebraic_operator
using WriteVTK
using LinearAlgebra: tr, inv, det
using LineSearches: BackTracking
import GridapODEs.TransientFETools: ∂t

# ## Problem setting

# ### Domain
model = DiscreteModelFromFile("../runs/elasticFlag_coarse.json")
writevtk(model,"model")

# This will produce an output in which we can identify the different parts of the domain, with the associated labels and tags.
#
# | Part | Notation | Label | Tag |
# | :---: | :---:| :---: | :---: |
# | Solid-cylinder wall | $\Gamma_{\rm S,D_{cyl}}$ | "fixed" | 1 |
# | Fluid-solid interface | $\Gamma_{\rm FS}$ | "interface" | 2 |
# | Channel inlet | $\Gamma_{\rm F,D_{in}}$ | "inlet" | 3 |
# | Channel outlet | $\Gamma_{\rm F,N_{out}}$ | "outlet" | 4 |
# | Channel walls | $\Gamma_{\rm F,D_{wall}}$ | "noSlip" | 5 |
# | Fluid-cylinder wall | $\Gamma_{\rm F,D_{cyl}}$ | "cylinder" | 6 |
# | Fluid domain | $\Omega_{\rm F}$ | "fluid" | 7 |
# | Solid domain | $\Omega_{\rm S}$ | "solid" | 8 |
#
# ![](../assets/fsi/tags.png)

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

# ### Source terms
hN(x) = VectorValue( 0.0, 0.0 )
f(x) = VectorValue( 0.0, 0.0 )
s(x) = VectorValue( 0.0, 0.0 )

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

# ## Numerical scheme
# ### FE Spaces
model_solid = DiscreteModel(model,"solid")
model_fluid = DiscreteModel(model,"fluid")

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
Vf = TestFESpace(
    model=model_fluid,
    valuetype=VectorValue{2,Float64},
    reffe=:Lagrangian,
    order=k,
    conformity =:H1,
    dirichlet_tags=["inlet", "noSlip", "cylinder","interface"])
Q = TestFESpace(
		model=model,
		valuetype=Float64,
		order=k-1,
		reffe=:Lagrangian,
		conformity=:C0)
Qf = TestFESpace(
  model=model_fluid,
  valuetype=Float64,
  order=k-1,
  reffe=:Lagrangian,
  conformity=:C0)

U_u = TransientTrialFESpace(Vu,[u_noSlip, u_noSlip, u_noSlip, u_noSlip, u_noSlip])
U_v = TransientTrialFESpace(Vv,[u_in, u_noSlip, u_noSlip, u_noSlip])
P = TrialFESpace(Q)
U0_f = TrialFESpace(Vf, [u_in(t0), u_noSlip(t0), u_noSlip(t0), u_noSlip(t0)])
Pf = TrialFESpace(Qf)

Y = MultiFieldFESpace([Vu,Vv,Q])
X = MultiFieldFESpace([U_u,U_v,P])
Y0 = MultiFieldFESpace([Vf,Qf])
X0 = MultiFieldFESpace([U0_f,Pf])

# ### Weak form
function a_ϕ_Ωf(x,y)
    u,v,p = x
    ϕ,φ,q = y
    inner( σ_m(ε(u)), ∇(ϕ) )
end
function a_ϕ_Γi(x,y)
		u_Γ,v_Γ,p_Γ = x
    ϕ_Γ,φ_Γ,q_Γ = y
    uf = jump(u_Γ)
    ϕf = jump(ϕ_Γ)
    - (ϕf *  n_Γi*σ_m(ε(uf)))
end
function a_ϕ_Ωs(x,xt,y)
    u,v,p = x
    ut,vt,pt = xt
    ϕ,φ,q = y
    inner( ut - v, ϕ )
end
function da_dut_ϕ_Ωs(dxt,y)
    dut,dvt,dpt = dxt
    ϕ,φ,q = y
    inner( dut, ϕ )
end
function da_dv_ϕ_Ωs(dx,y)
    du,dv,dp = dx
    ϕ,φ,q = y
    -inner( dv, ϕ )
end
function a_φ_Ωf(x,xt,y)
    u,v,p = x
    ut,vt,pt = xt
    ϕ,φ,q = y
		inner( J(∇(u)) * ρ_f * vt, φ ) + φ * ( J(∇(u)) * ρ_f * Finv(∇(u)) * conv(v-ut,∇(v)) ) + inner( J(∇(u)) * σ_dev(∇(v),Finv(∇(u))) * FinvT(∇(u)), ∇(φ) ) - (∇*φ)* J(∇(u)) * p * tr(FinvT(∇(u)))
end
function da_du_φ_Ωf(x,xt,dx,y)
    u,v,p = x
    ut,vt,pt = xt
    du,dv,dp = dx
    ϕ,φ,q = y
		inner( dJ(∇(u),∇(du)) * ρ_f * vt, φ ) +	φ * ρ_f * ( dJ(∇(u),∇(du)) * Finv(∇(u)) * conv(v-ut,∇(v)) + J(∇(u)) * dFinv(∇(u),∇(du)) * conv(v-ut,∇(v)) ) + inner( dJ(∇(u),∇(du)) * σ_dev(∇(v),Finv(∇(u))) * FinvT(∇(u)) + J(∇(u)) * σ_dev(∇(v),dFinv(∇(u),∇(du))) * FinvT(∇(u)) + J(∇(u)) * σ_dev(∇(v),Finv(∇(u))) * dFinvT(∇(u),∇(du)) , ∇(φ) ) + (∇*φ) * p * ( dJ(∇(u),∇(du)) * tr(FinvT(∇(u))) + J(∇(u)) * tr(dFinvT(∇(u),∇(du))) )
end
function da_dut_φ_Ωf(x,dxt,y)
    u,v,p = x
    dut,dvt,dpt = dxt
    ϕ,φ,q = y
		- φ * ( J(∇(u)) * ρ_f * Finv(∇(u)) * conv(dut,∇(v)) ) 
end
function da_dv_φ_Ωf(x,xt,dx,y)
    u,v,p = x
    ut,vt,pt = xt
    du,dv,dp = dx
    ϕ,φ,q = y
		φ * ( J(∇(u)) * ρ_f * Finv(∇(u)) * dconv(dv,∇(dv),v-ut,∇(v)) ) + inner( J(∇(u)) * σ_dev(∇(dv),Finv(∇(u))) * FinvT(∇(u)), ∇(φ) )
end
function da_dvt_φ_Ωf(x,dxt,y)
    u,v,p = x
    dut,dvt,dpt = dxt
    ϕ,φ,q = y
		inner( J(∇(u)) * ρ_f * dvt, φ )
end
function da_dp_φ_Ωf(x,dx,y)
    u,v,p = x
    du,dv,dp = dx
    ϕ,φ,q = y
		- (∇*φ)* J(∇(u)) * dp * tr(FinvT(∇(u)))
end
function a_φ_Ωs(x,xt,y)
    u,v = x
    ut,vt = xt
    ϕ,φ = y
		inner( ρ_s * vt, φ ) + inner( F(∇(u)) * S(∇(u)), ∇(φ) )
end
function da_du_φ_Ωs(x,dx,y)
    u,v = x
    du,dv = dx
    ϕ,φ = y
		inner( dF(∇(du)) * S(∇(u)) + F(∇(u)) * dS(∇(u),∇(du)), ∇(φ) )
end
function da_dvt_φ_Ωs(x,dxt,y)
    u,v = x
    dut,dvt = dxt
    ϕ,φ = y
		inner( ρ_s * dvt, φ )
end
function a_q_Ωf(x,xt,y)
    u,v,p = x
    ut,vt,pt = xt
    ϕ,φ,q = y
		println("--- XX --- ")
		#q * (∇*(J(∇(u))*Finv(∇(u))*v))
		#q * ( ∇(J(∇(u)))*Finv(∇(u))*v + J(∇(u))*(∇*Finv(∇(u)))*v + J(∇(u))*inner(Finv(∇(u))*(∇(v)')) )
		q * ( ∇*v )
end
function da_du_q_Ωf(x,dx,y)
    u,v,p = x
    du,dv,dp = dx
    ϕ,φ,q = y
		#q * (∇*((dJ(∇(u),∇(du))*Finv(∇(u))+J(∇(u))*dFinv(∇(u),∇(du)))*v))
		#q * ( ∇(dJ(∇(u),∇(du)))*Finv(∇(u))*v + ∇(J(∇(u)))*dFinv(∇(u),∇(du))*v  + dJ(∇(u),∇(du))*(∇*Finv(∇(u)))*v + J(∇(u))*(∇*dFinv(∇(u),∇(du)))*v + dJ(∇(u),∇(du))*Finv(∇(u))*(∇(v)') + J(∇(u))*dFinv(∇(u),∇(du))*(∇(v)') )
end
function da_dv_q_Ωf(x,dx,y)
    u,v,p = x
    du,dv,dp = dx
    ϕ,φ,q = y
		#q * (∇*(J(∇(u))*Finv(∇(u))*dv))
		#q * ( ∇(J(∇(u)))*Finv(∇(u))*dv + J(∇(u))*(∇*Finv(∇(u)))*dv + J(∇(u))*Finv(∇(u))*(∇(dv)') )
		q * ∇*dv
end

function res_Ωf(t,x,xt,y)
		a_ϕ_Ωf(x,y) #+ a_φ_Ωf(x,xt,y) + a_q_Ωf(x,xt,y)
end
function res_Ωs(t,x,xt,y)
		a_ϕ_Ωs(x,xt,y) + a_φ_Ωs(x,xt,y)
end

function jac_Ωf(t,x,xt,dx,y)
    a_ϕ_Ωf(dx,y) #+ da_du_φ_Ωf(x,xt,dx,y) + da_dv_φ_Ωf(x,xt,dx,y) + da_dp_φ_Ωf(x,dx,y) + da_dv_q_Ωf(x,dx,y) #+ da_du_q_Ωf(x,dx,y)
end
function jac_Ωs(t,x,xt,dx,y)
    da_dv_ϕ_Ωs(dx,y) #+ da_du_φ_Ωs(x,dx,y)
end

function jac_t_Ωf(t,x,xt,dxt,y)
		da_dut_φ_Ωf(x,dxt,y) #+ da_dvt_φ_Ωf(x,dxt,y)
end
function jac_t_Ωs(t,x,xt,dxt,y)
		da_dut_ϕ_Ωs(dxt,y) #+ da_dvt_φ_Ωs(x,dxt,y)
end

function a_Stokes(x, y)
    v, p = x
    φ, q = y
    inner(ε(φ), 2*μ_f*(ε(v))) - (∇ * φ) * p + q * (∇ * v)
end

# ### Numerical integration
trian_fluid = Triangulation(model_fluid)
trian_solid = Triangulation(model_solid)
trian_Γi = InterfaceTriangulation(model_fluid,model_solid)
n_Γi = get_normal_vector(trian_Γi)

degree = 2*k
quad_fluid = CellQuadrature(trian_fluid,degree)
quad_solid = CellQuadrature(trian_solid,degree)
idegree = 2*k
quad_Γi = CellQuadrature(trian_Γi,idegree)

# ### Algebraic System of Equations
#t_Ωf = FETerm(res_Ωf,jac_Ωf,,trian_fluid,quad_fluid)
t_Ωf = FETerm(res_Ωf,jac_Ωf,jac_t_Ωf,trian_fluid,quad_fluid)
t_Ωs = FETerm(res_Ωs,jac_Ωs,jac_t_Ωs,trian_solid,quad_solid)
t_Γi = LinearFETerm(a_ϕ_Γi,trian_Γi,quad_Γi)
t_Stokes_Ω = LinearFETerm(a_Stokes, trian_fluid, quad_fluid)
op = TransientFEOperator(X,Y,t_Ωf)#,t_Ωs)#,t_Γi)
op_Stokes = FEOperator(X0,Y0,t_Stokes_Ω)

# ### Solution
# Solve Stokes to get initial solution
xhSt = solve(op_Stokes)

# # ## PostProcessing
# # Write to vtk
# function writeStokesPVD(filePath::String, trian::Triangulation, sol; append=false)
#     outfiles = paraview_collection(filePath, append=append) do pvd
#         for (i, (xh, t)) in enumerate(sol)
#             uh = xh.blocks[1]
# 						uhf = restrict(uh,trian)
#             ph = xh.blocks[2]
# 						phf = restrict(ph,trian)
#             pvd[t] = createvtk(
#                 trian,
#                 filePath * "_st_$t.vtu",
#                 cellfields = ["uh" => uhf, "ph" => phf],
#             )
#         end
#     end
# end
# function write_NS_f_PVD(filePath::String, trian::Triangulation, sol; append=false)
# 		println("XXX")
# 		println(typeof(sol))
#     outfiles = paraview_collection(filePath, append=append) do pvd
#         for (i, (xh, t)) in enumerate(sol)
# 						println(typeof(xh))
#             uh = xh.blocks[1]
# 						uhf = restrict(uh,trian)
# 						println(length(uhf))
#             vh = xh.blocks[2]
# 						vhf = restrict(vh,trian)
# 						println(length(vhf))
#             ph = xh.blocks[3]
# 						phf = restrict(ph,trian)
# 						println(length(phf))
#             pvd[t] = createvtk(
#                 trian,
#                 filePath * "_f_$t.vtu",
#                 cellfields = ["uh" => uhf, "vh" => vhf, "ph" => phf],
#             )
#         end
#     end
# end
# function write_NS_s_PVD(filePath::String, trian::Triangulation, sol; append=false)
# 		println("XXX")
# 		println(typeof(sol))
#     outfiles = paraview_collection(filePath, append=append) do pvd
#         for (i, (xh, t)) in enumerate(sol)
# 						println(typeof(xh))
#             uh = xh.blocks[1]
# 						uhf = restrict(uh,trian)
# 						println(length(uhf))
#             vh = xh.blocks[2]
# 						vhf = restrict(vh,trian)
# 						println(length(vhf))
#             pvd[t] = createvtk(
#                 trian,
#                 filePath * "_s_$t.vtu",
#                 cellfields = ["uh" => uhf, "vh" => vhf, "ph" => phf],
#             )
#         end
#     end
# end

# Initialize Paraview files
folderName = "fsi-results"
fileName = "fields"
if !isdir(folderName)
    mkdir(folderName)
end
filePath = join([folderName, fileName], "/")
uh = restrict(xhSt.blocks[1],trian_fluid)
ph = restrict(xhSt.blocks[2],trian_fluid)
writevtk(trian_fluid, filePath * "_st_0.0.vtu",cellfields = ["uh" => uh, "ph" => ph])
#writeStokesPVD(filePath, trian_fluid, [(xhSt, 0.0)])

U0_u = U_u(0.0)
uh0 = interpolate_everywhere(U0_u, u_noSlip(0.0))
vh0 = xhSt[1]
ph0 = xhSt[2]
xh0 = Gridap.MultiField.MultiFieldFEFunction(X(0.0),[uh0,vh0,ph0])
uhf = restrict(xh0.blocks[1],trian_fluid)
vhf = restrict(xh0.blocks[2],trian_fluid)
phf = restrict(xh0.blocks[3],trian_fluid)
writevtk(trian_fluid, filePath * "_f0_0.0.vtu",cellfields = ["uh" => uhf, "vh" => vhf, "ph" => phf])
uhs = restrict(xh0.blocks[1],trian_solid)
vhs = restrict(xh0.blocks[2],trian_solid)
writevtk(trian_solid, filePath * "_s0_0.0.vtu",cellfields = ["uh" => uhs, "vh" => vhs])

# # FSI solution
 nls = NLSolver(show_trace = false,
							 method = :newton,
							 linesearch = BackTracking(),
							 )
odes =  ThetaMethod(nls, 0.001, 0.5)
solver = TransientFESolver(odes)
sol_t = solve(solver, op, xh0, 0.0, 0.002)
println(typeof(sol_t))
for (xh,t) in sol_t
		println(typeof(t))
end
# write_NS_f_PVD(filePath, trian_fluid, sol_t, append=true)
# write_NS_s_PVD(filePath, trian_solid, sol_t, append=true)


# ### Quantities of Interest
# trian_ΓS = BoundaryTriangulation(model,["cylinder","interface"])
# quad_ΓS = CellQuadrature(trian_ΓS,bdegree)
# n_ΓS = get_normal_vector(trian_ΓS)
# uh_ΓS = restrict(uhf_fluid,trian_ΓS)
# ph_ΓS = restrict(ph_fluid,trian_ΓS)
# FD, FL = sum( integrate( (σ_dev_f(ε(uh_ΓS))*n_ΓS - ph_ΓS*n_ΓS), trian_ΓS, quad_ΓS ) )
# println("Drag force: ", FD)
# println("Lift force: ", FL)

