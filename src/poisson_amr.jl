#
# In this tutorial, we will learn:
#
#    - How to use adaptive mesh refinement (AMR) with Gridap
#    - How to set up a Poisson problem on an L-shaped domain
#    - How to implement error estimation and marking strategies
#    - How to visualize the AMR process and results
#
# ## Problem Overview
#
# We will solve the Poisson equation on an L-shaped domain using adaptive mesh refinement.
# The L-shaped domain is known to introduce a singularity at its reentrant corner,
# making it an excellent test case for AMR. The problem is:
#
# ```math
# \begin{aligned}
# -\Delta u &= f  &&\text{in } \Omega \\
# u &= g &&\text{on } \partial\Omega
# \end{aligned}
# ```
#
# where Ω is the L-shaped domain, and we choose an exact solution with a singularity
# to demonstrate the effectiveness of AMR.
#
# ## Error Estimation and AMR Process
#
# We use a residual-based a posteriori error estimator η. For each element K in the mesh,
# the local error indicator ηK is computed as:
#
# ```math
# \eta_K^2 = h_K^2\|f + \Delta u_h\|_{L^2(K)}^2 + h_K\|\jump{\nabla u_h \cdot n}\|_{L^2(\partial K)}^2
# ```
#
# where:
# - h_K is the diameter of element K
# - u_h is the computed finite element solution
# - The first term measures the element residual
# - The second term measures the jump in the normal derivative across element boundaries
#
# The AMR process follows these steps in each iteration:
#
# 1. **Solve**: Compute the finite element solution u_h on the current mesh
# 2. **Estimate**: Calculate error indicators ηK for each element
# 3. **Mark**: Use Dörfler marking to select elements for refinement
#    - Sort elements by error indicator
#    - Mark elements containing a fixed fraction (here 80%) of total error
# 4. **Refine**: Refine selected elements to obtain a new mesh. In this example, 
#    we will be using the newest vertex bisection (NVB) method to keep the mesh 
#    conforming (without any hanging nodes).
#
# This adaptive loop continues until either:
# - A maximum number of iterations is reached
# - The estimated error falls below a threshold
# - The solution achieves desired accuracy
#
# The process automatically concentrates mesh refinement in regions of high error,
# particularly around the reentrant corner where the solution has reduced regularity.
# This results in better accuracy per degree of freedom compared to uniform refinement.
#
# ## Required Packages

using Gridap, Gridap.Geometry, Gridap.Adaptivity
using DataStructures

# ## Problem Setup
#
# We define an exact solution that contains a singularity at the corner (0.5, 0.5)
# of the L-shaped domain. This singularity will demonstrate how AMR automatically
# refines the mesh in regions of high error.

ϵ = 1e-2
r(x) = ((x[1]-0.5)^2 + (x[2]-0.5)^2)^(1/2)
u_exact(x) = 1.0 / (ϵ + r(x))

# Create an L-shaped domain by removing a quadrant from a unit square.
# The domain is [0,1]² \ [0.5,1]×[0.5,1]
function LShapedModel(n)
  model = CartesianDiscreteModel((0,1,0,1),(n,n))
  cell_coords = map(mean,get_cell_coordinates(model))
  l_shape_filter(x) = (x[1] < 0.5) || (x[2] < 0.5)
  mask = map(l_shape_filter,cell_coords)
  return simplexify(DiscreteModelPortion(model,mask))
end

# Define the L2 norm for error estimation.
# These will be used to compute both local and global error measures.
l2_norm(he,xh,dΩ) = ∫(he*(xh*xh))*dΩ
l2_norm(xh,dΩ) = ∫(xh*xh)*dΩ

# ## AMR Step Function
#
# The `amr_step` function performs a single step of the adaptive mesh refinement process:
# 1. Solves the Poisson problem on the current mesh
# 2. Estimates the error using residual-based error indicators
# 3. Marks cells for refinement using Dörfler marking
# 4. Refines the mesh using newest vertex bisection (NVB)

function amr_step(model,u_exact;order=1)
  # Create FE spaces with Dirichlet boundary conditions on all boundaries
  reffe = ReferenceFE(lagrangian,Float64,order)
  V = TestFESpace(model,reffe;dirichlet_tags=["boundary"])
  U = TrialFESpace(V,u_exact)
  
  # Setup integration measures
  Ω = Triangulation(model)
  Γ = Boundary(model)
  Λ = Skeleton(model)
  
  dΩ = Measure(Ω,4*order)
  dΓ = Measure(Γ,2*order)
  dΛ = Measure(Λ,2*order)
  
  # Compute cell sizes for error estimation
  hK = CellField(sqrt.(collect(get_array(∫(1)dΩ))),Ω)

  # Get normal vectors for boundary and interface terms
  nΓ = get_normal_vector(Γ)
  nΛ = get_normal_vector(Λ)

  # Define the weak form
  ∇u(x)  = ∇(u_exact)(x)
  f(x)   = -Δ(u_exact)(x)
  a(u,v) = ∫(∇(u)⋅∇(v))dΩ
  l(v)   = ∫(f*v)dΩ
  
  # Define the residual error estimator
  # It includes volume residual, boundary jump, and interface jump terms
  ηh(u)  = l2_norm(hK*(f + Δ(u)),dΩ) +           # Volume residual
           l2_norm(hK*(∇(u) - ∇u)⋅nΓ,dΓ) +       # Boundary jump
           l2_norm(jump(hK*∇(u)⋅nΛ),dΛ)          # Interface jump
  
  # Solve the FE problem
  op = AffineFEOperator(a,l,U,V)
  uh = solve(op)
  
  # Compute error indicators
  η = estimate(ηh,uh)
  
  # Mark cells for refinement using Dörfler marking
  # This strategy marks cells containing a fixed fraction (0.8) of the total error
  m = DorflerMarking(0.8)
  I = Adaptivity.mark(m,η)
  
  # Refine the mesh using newest vertex bisection
  method = Adaptivity.NVBRefinement(model)
  amodel = refine(method,model;cells_to_refine=I)
  fmodel = Adaptivity.get_model(amodel)

  # Compute the global error for convergence testing
  error = sum(l2_norm(uh - u_exact,dΩ))
  return fmodel, uh, η, I, error
end

# ## Main AMR Loop
#
# We perform multiple AMR steps, refining the mesh iteratively and solving
# the problem on each refined mesh. This demonstrates how the error decreases
# as the mesh is adaptively refined in regions of high error.

nsteps = 5
order = 1
model = LShapedModel(10)

last_error = Inf
for i in 1:nsteps
  # Perform one AMR step
  fmodel, uh, η, I, error = amr_step(model,u_exact;order)
  
  # Create indicator field for refined cells
  is_refined = map(i -> ifelse(i ∈ I, 1, -1), 1:num_cells(model))
  
  # Visualize results
  Ω = Triangulation(model)
  writevtk(
    Ω,"model_$(i-1)",append=false,
    cellfields = [
      "uh" => uh,                    # Computed solution
      "η" => CellField(η,Ω),        # Error indicators
      "is_refined" => CellField(is_refined,Ω),  # Refinement markers
      "u_exact" => CellField(u_exact,Ω),       # Exact solution
    ],
  )
  
  # Print error information and verify convergence
  println("Error: $error, Error η: $(sum(η))")
  @test (i < 3) || (error < last_error)
  last_error = error
  model = fmodel
end

# ## Conclusion
#
# In this tutorial, we have demonstrated how to:
# 1. Implement adaptive mesh refinement for the Poisson equation
# 2. Use residual-based error estimation to identify regions needing refinement
# 3. Apply Dörfler marking to select cells for refinement
# 4. Visualize the AMR process and solution convergence
#
# The results show how AMR automatically refines the mesh near the singularity,
# leading to more efficient and accurate solutions compared to uniform refinement.
