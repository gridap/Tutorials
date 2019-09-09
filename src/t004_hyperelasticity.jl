# # Tutorial 4: Hyperelasticity
#
#md # [![](https://mybinder.org/badge_logo.svg)](@__BINDER_ROOT_URL__/notebooks/t004_hyperelasticity.ipynb)
#md # [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/notebooks/t004_hyperelasticity.ipynb)
# 
# ## Problem statement

using Gridap
using LinearAlgebra

# Model
model = CartesianDiscreteModel(
  domain=(0.0,0.1,0.0,1.0), partition=(3,20))

# Construct the FEspace
order = 1
diritags = [1,2,5,3,4,6]
T = VectorValue{2,Float64}
fespace = CLagrangianFESpace(T,model,order,diritags)

g0(x) = zero(T)
g1(x) = VectorValue(0.0,-0.03)
V = TestFESpace(fespace)
U = TrialFESpace(fespace,[g0,g0,g0,g1,g1,g1])

# Setup integration
trian = Triangulation(model)
quad = CellQuadrature(trian,order=2)

# Material parameters
const λ = 100.0
const μ = 1.0

# Identity tensor
const I = one(TensorValue{2,Float64,4})

# Deformation Gradient
F(∇u) = I + ∇u'

J(F) = sqrt(det(C(F)))

#Green strain

#E(F) = 0.5*( F'*F - I )

@law dE(x,∇du,∇u) = 0.5*( ∇du*F(∇u) + (∇du*F(∇u))' )

# Right Cauchy-green deformation tensor

C(F) = (F')*F

# Constitutive law (Neo hookean)

@law function S(x,∇u)
  Cinv = inv(C(F(∇u)))
  μ*(I-Cinv) + λ*log(J(F(∇u)))*Cinv
end

@law function dS(x,∇du,∇u)
  Cinv = inv(C(F(∇u)))
  _dE = dE(x,∇du,∇u)
  λ*inner(Cinv,_dE)*Cinv + 2*(μ-λ*log(J(F(∇u))))*Cinv*_dE*(Cinv')
end

# Cauchy stress tensor

@law σ(x,∇u) = (1.0/J(F(∇u)))*F(∇u)*S(x,∇u)*(F(∇u))'

@law σ_lin(x,ε) = λ*trace(ε)*one(ε) + 2*μ*ε #TODO trace

# Weak form

res(u,v) = inner( dE(∇(v),∇(u)) , S(∇(u)) )

jac_mat(u,v,du) = inner( dE(∇(v),∇(u)), dS(∇(du),∇(u)) )

jac_geo(u,v,du) = inner( ∇(v), S(∇(u))*∇(du) )

jac(u,v,du) = jac_mat(u,v,du) + jac_geo(u,v,du)

t_Ω = NonLinearFETerm(res,jac,trian,quad)

# FE problem
op = NonLinearFEOperator(V,U,t_Ω)

using NLsolve
using Gridap.FEOperators: NonLinearOpFromFEOp

alg_op = NonLinearOpFromFEOp(op)

f!(r,x) = residual!(r,alg_op,x)
j!(j,x) = jacobian!(j,alg_op,x)

x0 = 0.001*rand(Float64,num_free_dofs(U))
f0 = residual(alg_op,x0)
j0 = jacobian(alg_op,x0)

df = OnceDifferentiable(f!,j!,x0,f0,j0)

r = nlsolve(df,x0,show_trace=true)

uh = FEFunction(U,r.zero)

writevtk(trian,"results",cellfields=["uh"=>uh])



## Define the FESolver
#ls = LUSolver()
#tol = 1.e-3
#maxiters = 30
#nls = NewtonRaphsonSolver(ls,tol,maxiters)
#solver = NonLinearFESolver(nls)

## Solve!
#free_vals = 0.001*rand(Float64,num_free_dofs(U))
#uh = FEFunction(U,free_vals)
#solve!(uh,solver,op)
#
#writevtk(trian,"results",nref=3,cellfields=[
#  "uh"=>uh,"sigma"=>σ(∇(uh)),"S"=>S(∇(uh)),
#  "epsi"=>ε(uh),"dE"=>dE(∇(uh),∇(uh)),"sigma_lin"=>σ_lin(ε(uh))])

### Check derivatives
##
##d = 0.0000001
##∇u = TensorValue(1.0,1.0,2.0,3.0)
##∇du = TensorValue(0.0,1.0,2.0,1.0)
##x = zero(T)
##
##S1 = S(x,∇u+d*∇du)
##S2 = S(x,∇u) + d*dS(x,∇du,∇u)
##
##@show S1
##@show S2
##@show S1-S2

#d = 0.00001
#
#uh_vals = 0.1*rand(Float64,num_free_dofs(U))
#uh = FEFunction(U,uh_vals)
#
#duh_vals = d*0.01*rand(Float64,num_free_dofs(U))
#duh = FEFunction(V,duh_vals)
#
#r1 = residual(op,uh.cellfield+duh) #TODO
#
#r2 = residual(op,uh) + jacobian(op,uh)*duh_vals
#
#er = r1 - r2
#
#@show r1
#@show r2
#@show er
#
#eh = FEFunction(V,er)
#
#using Gridap.FEOperators: NonLinearOpFromFEOp
#
#alg_op = NonLinearOpFromFEOp(op)
#
#residual!(r1,alg_op,uh_vals+duh_vals)
#
#residual!(r2,alg_op,uh_vals)
#
#r2 = r2 + jacobian(alg_op,uh_vals)*duh_vals
#
#er = r1 - r2
#
#eh = FEFunction(V,er)
#
#@show r1
#@show r2
#@show er
#
#writevtk(trian,"results",cellfields=["eh"=>eh])

