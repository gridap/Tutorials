module TransientHeatEq

using Gridap

model = DiscreteModelFromFile("../models/model.json")
labels = FaceLabels(model)
add_tag_from_tags!(labels,"diri0",["sides", "sides_c"])

trian = Triangulation(model)
quad = CellQuadrature(trian,degree=2)

neumanntags = ["circle", "triangle", "square"]
btrian = BoundaryTriangulation(model,neumanntags)
bquad = CellQuadrature(btrian,degree=2)

V = FESpace(
  reffe=:Lagrangian,
  conformity=:H1,
  valuetype=Float64,
  order = 1,
  model = model,
  labels = labels,
  diritags = "diri0");

V0 = TestFESpace(V)
U0 = TrialFESpace(V,0.0)

# Time-dependent flux
function h(t)
  s = -1.0
  tc = 0.5
  if t < 0.0
    return zero(s*tc)
  elseif t < tc
    return s*t
  else
    return s*tc
  end
end

function bt(dt,t,u0)
  b_Ω(v) = (1/dt)*inner(v,u0)
  ht = h(t)
  b_Γ(v) = inner(v, (x)->ht)
  t_Ω = FESource(b_Ω,trian,quad)
  t_Γ = FESource(b_Γ,btrian,bquad)
  NonLinearFEOperator(V0,U0,t_Ω,t_Γ) # Un-intuitive
end

function run(Tf,Nt)

  dt = Tf / Nt

  # Prepare matrix for all time steps
  k = 1.0
  a_Ω(v,u) = (1/dt)*inner(v,u) + inner(∇(v), k*∇(u))
  t_Ω = LinearFETerm(a_Ω,trian,quad)
  op = LinearFEOperator(V0,U0,t_Ω) 
  
  # Factorize matrix for all time steps
  ls = LUSolver()
  K = op.mat
  ss = symbolic_setup(ls,K)
  ns = numerical_setup(ss,K)

  # Allocate memory
  x0 = zeros(num_free_dofs(U0))
  x1 = zeros(num_free_dofs(U0))
  f = op.vec
  
  for nt in 1:Nt
  
    t = dt*nt

    # Assemble rhs
    u0 = FEFunction(U0,x0)
    fop = bt(dt,t,u0)
    residual!(f,fop,u0) # Un-intuitive

    # Forward substitution
    solve!(x1,ns,f)

    # Write time step
    u1 = FEFunction(U0,x1)
    writevtk(trian,"results_$(lpad(nt,3,'0'))",cellfields=["uh"=>u1])
    if nt == 1
      writevtk(trian,"results_$(lpad(0,3,'0'))",cellfields=["uh"=>u0])
    end

    # Update
    x0 = x1

  end

end

Tf = 1.5
Nt = 60

# Do the work!
run(Tf,Nt)

end # module
