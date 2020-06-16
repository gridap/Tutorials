#md # !!! note
#
#     This tutorial is under construction, but the code below is already functional.
#
# 
# ## Problem statement

using Gridap
using LinearAlgebra: inv, det
using LineSearches: BackTracking

# Material parameters
const λ = 100.0
const μ = 1.0

# Deformation Gradient
F(∇u) = one(∇u) + ∇u'

J(F) = sqrt(det(C(F)))

#Green strain

#E(F) = 0.5*( F'*F - one(F) )

@law dE(∇du,∇u) = 0.5*( ∇du⋅F(∇u) + (∇du⋅F(∇u))' )

# Right Cauchy-green deformation tensor

C(F) = (F')⋅F

# Constitutive law (Neo hookean)

@law function S(∇u)
  Cinv = inv(C(F(∇u)))
  μ*(one(∇u)-Cinv) + λ*log(J(F(∇u)))*Cinv
end

@law function dS(∇du,∇u)
  Cinv = inv(C(F(∇u)))
  _dE = dE(∇du,∇u)
	λ*(Cinv⊙_dE)*Cinv + 2*(μ-λ*log(J(F(∇u))))*Cinv⋅_dE⋅(Cinv')
end

# Cauchy stress tensor

@law σ(∇u) = (1.0/J(F(∇u)))*F(∇u)⋅S(∇u)⋅(F(∇u))'

# Weak form

res(u,v) = dE(∇(v),∇(u)) ⊙ S(∇(u))

jac_mat(u,du,v) =  dE(∇(v),∇(u)) ⊙ dS(∇(du),∇(u))

jac_geo(u,du,v) = ∇(v) ⊙ ( S(∇(u))⋅∇(du) )

jac(u,du,v) = jac_mat(u,v,du) + jac_geo(u,v,du)

# Model
domain = (0,1,0,1)
partition = (20,20)
model = CartesianDiscreteModel(domain,partition)

# Define new boundaries
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"diri_0",[1,3,7])
add_tag_from_tags!(labels,"diri_1",[2,4,8])

# Construct the FEspace
V = TestFESpace(
  model=model,valuetype=VectorValue{2,Float64},
  reffe=:Lagrangian,conformity=:H1,order=1,
  dirichlet_tags = ["diri_0", "diri_1"])

# Setup integration
trian = Triangulation(model)
degree = 2
quad = CellQuadrature(trian,degree)

# Setup weak form terms
t_Ω = FETerm(res,jac,trian,quad)

# Setup non-linear solver
nls = NLSolver(
  show_trace=true,
  method=:newton,
  linesearch=BackTracking())

solver = FESolver(nls)

function run(x0,disp_x,step,nsteps)

  g0 = VectorValue(0.0,0.0)
  g1 = VectorValue(disp_x,0.0)
  U = TrialFESpace(V,[g0,g1])

  #FE problem
  op = FEOperator(U,V,t_Ω)
  
  println("\n+++ Solving for disp_x $disp_x in step $step of $nsteps +++\n")
  
  uh = FEFunction(U,x0)

  uh, = solve!(uh,solver,op)
  
  writevtk(trian,"results_$(lpad(step,3,'0'))",cellfields=["uh"=>uh,"sigma"=>σ(∇(uh))])

  return get_free_values(uh)

end

function runs()

 disp_max = 0.75
 disp_inc = 0.02
 nsteps = ceil(Int,abs(disp_max)/disp_inc)
 
 x0 = zeros(Float64,num_free_dofs(V))

 for step in 1:nsteps
   disp_x = step * disp_max / nsteps
   x0 = run(x0,disp_x,step,nsteps)
 end

end

#Do the work!
runs()

# Picture of the last load step
# ![](../assets/hyperelasticity/neo_hook_2d.png)
#
# ##  Extension to 3D
# 
# Extending this tutorial to the 3D case is straightforward. It is leaved as an exercise.
#
# ![](../assets/hyperelasticity/neo_hook_3d.png)

