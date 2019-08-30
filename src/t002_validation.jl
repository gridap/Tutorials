# # Tutorial 2: Code validation
#
#md # [![](https://mybinder.org/badge_logo.svg)](@__BINDER_ROOT_URL__/notebooks/t002_validation.ipynb)
#md # [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/notebooks/t002_validation.ipynb)
# 
# ## Learning outcomes
#
# - How to implement the method of manufactured solutions
# - How to perform a convergence test
# - How to define the discretization error
# - How to integrate error norms
# - How to generate Cartesian meshes in arbitrary dimensions
#
# ## Problem statement
#
# In this tutorial, we show how to validate a code using the well known *method of manufactured solutions*. For the sake of simplicity, we consider the Poisson equation in the unit square $\Omega\doteq (0,1)^2$ as a model problem,
#
#
# ```math
# \left\lbrace
# \begin{aligned}
# -\Delta u = f  \ \text{in} \ \Omega\\
# u = g \ \text{on}\ \partial\Omega.\\
# \end{aligned}
# \right.
# ```
#
# We are going to consider two different manufactured solutions. On the one hand, we consider function $u(x)=x_1+x_2$, which can be exactly represented by the FE interpolation that we construct below. Thus, one expects that the obtained approximation error is near the machine precision. We are going to check that this is true in the code. On the other hand, we consider a function that cannot be captured exactly by the interpolation, namely $u(x)=x_2 \sin(2 \pi\ x_1)$. Here, our goal is to confirm that the convergence order of the discretization error is the optimal one.
#
#
# ## Manufactured solution
#
# We start by defining the manufactured solution $u(x) = x_1+x_2$ and the source term $f$ associated with it, namely $f\doteq-\Delta(x_1+x_2)=0$.

using Gridap

u(x) = x[1] + x[2]
f(x) = 0.0

# Note that it is important that function `f` returns a `Float64` value. This is needed since we are going to use `Float64` numbers to represent the solution.
#
# We also need to define the gradient of $u$ since we will compute the $H^1$ error norm later. In that case, the gradient is simply defined as
#

∇u(x) = VectorValue(1.0,1.0)

# Note that we have used the constructor `VectorValue` to build the vector that represents the gradient. However, we still need a final trick. We need to tell the Gridap library that the gradient of the function `u` is available in the function `∇u` (at this moment `u` and `∇u` are two standard Julia functions without any connection between them). This is done by adding an extra method to the function `gradient` (aka `∇`) defined in Gridap:

import Gridap: ∇
∇(::typeof(u)) = ∇u

# Now, it is possible to recover function `∇u` from function `u` as `∇(u)`. You can check that the following expression evaluates to `true`.

∇(u) === ∇u

# ## Cartesian mesh generation
#
# In order to discretized the geometry of the unit square, we use the Cartesian mesh generator available in Gridap:

limits = (0.0, 1.0, 0.0, 1.0)
model = CartesianDiscreteModel(domain=limits, partition=(4,4));

# The type `CartesianDiscreteModel` is a concrete type that inherits from `DiscreteModel`, which is specifically designed for building Cartesian meshes. The `CartesianDiscreteModel` constructor takes a tuple containing limits of the box we want to discretize  plus a tuple with the number of cells to be generated in each direction (here 4 by 4 cells). Note that the `CaresianDiscreteModel` is implemented for arbitrary dimensions. For instance, the following lines build a `CartesianDiscreteModel`  for the unit cube $(0,1)^3$ with 4 cells per direction

limits3d = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
model3d = CartesianDiscreteModel(domain=limits3d, partition=(4,4,4));

# You could also generate a mesh for the unit tesseract $(0,1)^4$ (i.e., the unit cube in 4D). Look how the 2D and 3D models are build and just follow the sequence.
#

# Let us return to the 2D `CartesianDiscreteModel` that we have already constructed. You can inspect it by writing it into vtk format. Note that you can also print a 3D model, but not a 4D one. In the future, it would be cool to generate a movie from a 4D model, but this functionality is not yet implemented.

writevtk(model,"model");


# If you open the generated files, you will see that the boundary vertices and facets are identified with the name "boundary". This is just what we need to impose the Dirichlet boundary conditions in this example.
#
# These are the vertices in the model
#
# ![](../assets/t002_validation/model_0.png)
#
# and these the facets
#
# ![](../assets/t002_validation/model_1.png)


# ## FE approximation
#
# We compute a FE approximation of the Poisson problem above by following the steps detailed in previous tutorial:

order = 1
diritag = "boundary"
V = CLagrangianFESpace(Float64,model,order,diritag)

V0 = TestFESpace(V)
U = TrialFESpace(V,u)

trian = Triangulation(model)
quad = CellQuadrature(trian,order=2)

a(v,u) = inner(∇(v), ∇(u))
b(v) = inner(v,f)

t_Ω = AffineFETerm(a,b,trian,quad)
op = LinearFEOperator(V0,U,t_Ω)

uh = solve(op);

# Note that we are imposing Dirichlet boundary conditions on the objects tagged as "boundary" and that we are using the manufactured solution `u` to construct the trial FE space. Not also that we are not explicitly constructing an `Assembler` object nor a `FESolver`. We are relying on default values.
#
#
# ## Measuring the discretization error
#
# Our goal is to check that the discratization error associated with the computed approximation `uh` is near machine precision. To this end, the first step is to compute the discretization error, which is done as you would expect:

e = u - uh;

# Once the error is defined, you can, e.g., visualize it.

writevtk(trian,"error",cellfields=["e" => e]);

# This generates a file called `error.vtu`. Open it with Paraview to check that the error is of the order of the machine precision.
#
# ![](../assets/t002_validation/error.png)
#
# A more rigorous way of quantifying the error is to measure it with a norm. Here, we use the $L^2$ and $H^1$ norms, which are defined as
#
# ```math
# \| w \|_{L^2}^2 \doteq \int_{\Omega} w^2 \ \text{d}\Omega, \quad 
# \| w \|_{H^1}^2 \doteq \int_{\Omega} w^2 + \nabla w \cdot \nabla w \ \text{d}\Omega.
#
# ```
#
# In order to compute these norms, we are going to use the `integrate` function. To this end, we need to define the integrands that we want to integrate, namely

l2(w) = inner(w,w)
h1(w) = a(w,w) + l2(w)

# Note that in order to define the integrand of the $H^1$ norm, we have reused function `a`, previously used to define the bilinear form of the problem.  Once we have defined the integrands, we are ready to compute the integrals. For the $L^2$ norm

el2 = sqrt(sum( integrate(l2(e),trian,quad) ))

# and for the $H^1$ norm

eh1 = sqrt(sum( integrate(h1(e),trian,quad) ))

# The `integrate` function works as follows. In the first argument, we pass the integrand. In the second and third arguments, we pass a `Triangulation` object and a`CellQuadrature` that represent the data needed in order to perform the integrals numerically. The `integrate` function returns an object containing the contribution to the integrated value of each cell in the given `Triangulation`. To end up with the desired error norms, one has to sum these contributions and take the square root. You can check that the computed error norms are close to machine precision (as one would expect).

tol = 1.e-10
@assert el2 < tol
@assert eh1 < tol


# ## Convergence test
#
# We end up this tutorial by performing a convergence test, where we are going to use all the new concepts we have learned.  We will consider a manufactured solution that does not belong to the FE interpolation space. In this test, we expect to see the optimal convergence order of the FE discretization.

# Here, we define the manufactured functions
const k = 2*pi
u(x) = sin(k*x[1]) * x[2]
∇u(x) = VectorValue(k*cos(k*x[1])*x[2], sin(k*x[1]))
f(x) = (k^2)*sin(k*x[1])*x[2]

# Since we have redefined the valiables `u`, `∇u`, and `f`, we need to execute these lines again

∇(::typeof(u)) = ∇u
b(v) = inner(v,f)

# In order to perform the convergence test, we write in a function all the code needed to perform a single computation and measure its error. The input of this function is the number of cells in each direction and the interpolation order. The output is the computed $L^2$ and $H^1$ error norms.

function run(n,order)

  limits = (0.0, 1.0, 0.0, 1.0)
  model = CartesianDiscreteModel(domain=limits, partition=(n,n))
  
  diritag = "boundary"
  V = CLagrangianFESpace(Float64,model,order,diritag)
  
  V0 = TestFESpace(V)
  U = TrialFESpace(V,u)
  
  trian = Triangulation(model)
  quad = CellQuadrature(trian,order=order+2)
  
  t_Ω = AffineFETerm(a,b,trian,quad)
  op = LinearFEOperator(V0,U,t_Ω)
  
  uh = solve(op)
  
  e = u - uh
  
  el2 = sqrt(sum( integrate(l2(e),trian,quad) ))
  eh1 = sqrt(sum( integrate(h1(e),trian,quad) ))

  (el2, eh1)

end

# The following function does the convergence test. It takes a vector of integers (representing the number of cells per direction in each computation) plus the interpolation order. It returns the $L^2$ and $H^1$ error norms for each computation as well as the corresponding cell size.

function conv_test(ns,order)

  el2s = Float64[]
  eh1s = Float64[]
  hs = Float64[]

  for n in ns

    el2, eh1 = run(n,order)
    h = 1.0/n

    push!(el2s,el2)
    push!(eh1s,eh1)
    push!(hs,h)

  end

  (el2s, eh1s, hs)

end

# We are ready to perform the test! We consider several mesh sizes and interpolation order equal to 2.

el2s, eh1s, hs = conv_test([8,16,32,64,128],2);

# With the generated data, we do the classical convergence plot.

using Plots

plot(hs,[el2s eh1s],
    xaxis=:log, yaxis=:log,
    label=["L2" "H1"],
    shape=:auto,
    xlabel="h",ylabel="error norm")

#src savefig("conv.png")

#md # If you run the code in a notebook, you will see a figure like this one:
#md # ![](../assets/t002_validation/conv.png)
#
#
# The generated curves make sense. It is observed that the convergence of the $H^1$ error is slower that $L^2$ one. However, in order to be more conclusive, we need to compute the slope of these lines. It can be done with this little function that internally uses a linear regression.

function slope(hs,errors)
  x = log10.(hs)
  y = log10.(errors)
  linreg = hcat(fill!(similar(x), 1), x) \ y
  linreg[2]
end

# The slope for the $L^2$ error norm is computed as

slope(hs,el2s)

# and for the $H^1$ error norm

slope(hs,eh1s)

#md # If your run these lines in a notebook, you will see that
#nb # As you can see,
# the slopes for the $L^2$ and $H^1$ error norms are circa 3 and 2 respectively (as one expects for interpolation order 2)
#
# Congrats, another tutorial done!


