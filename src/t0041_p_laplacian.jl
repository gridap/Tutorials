# # Tutorial 4: p-Laplacian
#
#md # [![](https://mybinder.org/badge_logo.svg)](@__BINDER_ROOT_URL__/notebooks/t0041_p_laplacian.ipynb)
#md # [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/notebooks/t0041_p_laplacian.ipynb)
# 
# ## Learning outcomes
#
# - How to solve a simple non-linear PDE in Gridap
# - How to define the weak residual and its Jacobian
# - How to setup and use a non-linear solver
# - How to define new boundaries from a given discrete model
# - How to interpolate a function in a FE space
#
# ## Problem statement
#
# The goal of this tutorial is to solve a non-linear PDEs in Gridap. For the sake of simplicity, we consider the [p-Laplacian](https://en.wikipedia.org/wiki/P-Laplacian) as the model problem. More complex PDEs will be considered in other tutorials. See, e.g., the tutorial on geometrically non-linear elasticity (hyper-elasticity) or the one on the incompressible Navier-Stokes equation. In this tutorial, the PDE we want to solve is
#
# ```math
# \left\lbrace
# \begin{aligned}
# -\nabla \cdot \left( |\nabla u|^{p-2}\ \nabla u \right) = f\ \text{in}\ \Omega\\
# u = 0 \ \text{on} \ \Gamma_0\\
# u = g \ \text{on} \ \Gamma_g\\
# \left( |\nabla u|^{p-2}\ \nabla u \right)\cdot n = 0 \ \text{on} \ \Gamma_{\rm N}
# \end{aligned}
# \right.
# ```
# This PDE is the p-Laplacian equation of degree $p>2$, equipped with homogeneous Dirichlet and Neumann boundary conditions on $\Gamma_0$ and $\Gamma_{\rm N}$  respectively, and in-homogeneous Dirichlet conditions on $\Gamma_g$.  The domain $\Omega$ is the one depicted in the figure below. The Dirichlet boundaries $\Gamma_0$ and $\Gamma_g$ are defined as the closure of the green and blue surfaces respectively, whereas the Neumann boundary is the remaining portion of the boundary $\Gamma_{\rm N}\doteq\partial\Omega \setminus (\Gamma_0\cup\Gamma_g)$. In this example, we consider the values $p=3$, $f=1$, and $g=2$.
#
# ![](../assets/t0041_p_laplacian/model.png)
#
#
# ## Numerical scheme
#
# As in previous tutorials, we discretize the problem with conforming Lagrangian FE spaces. For this formulation, the weak form reads: find $u\in U_g$ such that $[r(u)](v) = 0$ for all $v\in V_0$, where the weak residual $r: U_g \rightarrow (V_0)^\prime$ is defined as
# ```math
# [r(u)](v) \doteq \int_\Omega \nabla v \cdot \left( |\nabla u|^{p-2}\ \nabla u \right) \ {\rm d}\Omega - \int_\Omega v\ f \ {\rm d}\Omega.
# ```
# The spaces $U_g$ is the set of functions in $H^1(\Omega)$ that fulfill the Dirichlet boundary conditions, whereas $V_0$ is composed by functions in $H^1(\Omega)$ that vanish at the Dirichlet boundary.
#
# In order to solve this non-linear weak equation, we consider a Newton-Raphson method, which is associated with the following linearization of the problem:  $[r(u+\delta u)](v)\approx [r(u)](v) + [j(u)](v,\delta u)$. The Jacobian evaluated at $u\in U_g$ is the bilinear form defined as 
# ```math
# [j(u)](v,\delta u) \doteq \left.\dfrac{\rm d}{{\rm d} \varepsilon}\right|_{\varepsilon = 0} [r(u+\varepsilon \ \delta u)](v).
# ```
#
# For the current example, we have
#
# ```math
# [j(u)](v,\delta u) = \int_\Omega \nabla v \cdot \left( |\nabla u|^{p-2}\ \nabla \delta u \right) \ {\rm d}\Omega + (p-2) \int_\Omega \nabla v \cdot \left(  |\nabla u|^{p-4} (\nabla u \cdot \nabla \delta u) \nabla u  \right) \ {\rm d}\Omega.
# ```
#
# Note that the solution of this non-linear PDE with the Newton-Raphson method, will require to discretize both the residual $r$ and the Jacobian $j$. In Gridap, this is done by following an approach similar to the one already shown in previous tutorials for discretizing the bilinear and linear forms associated with linear FE problems. The specific details are discussed in next section.
#
# ## Implementation
#
# ### Defining new boundary identifiers
#
# The first step to solve this PDE in Gridap is to load the discretization of the computational domain. It that case we load the model from a file

using Gridap

model = DiscreteModelFromFile("../models/model.json");

# Once we have build the discrete model, we have to inspect it in order to see which boundaries are defined in it. To this end, write the model to vtk format and open the resulting file in paraview.
#

writevtk(model,"model");

# We need to impose Dirichlet boundary conditions on $\Gamma_0$ and $\Gamma_g$ as we have stated above, but non of these boundaries is identified in the model. E.g., you can easily see in paraview that the boundary identified as "sides" in the model only includes the vertices in the interior of $\Gamma_0$, but, in this example, we want to impose Dirichlet boundary conditions also on the vertices on the contour of $\Gamma_0$ for demonstration purposes. Fortunately, the objects on the contour of $\Gamma_0$ are identified in the model with the tag `"sides_c"` (see figure below). Thus, the Dirichlet boundary $\Gamma_0$ is build as the union of the objects identified as `"sides"` and `"sides_c"`.
#
# ![](../assets/t0041_p_laplacian/sides_c.png)
#
# Gridap provides a convenient way to create new object identifiers (referred as "tags") from existing ones. It is done as follows. First, we need to extract from the model, the object that holds the information about the boundary identifiers, which in Gridap is represented with the `FaceLabels` type:

labels = FaceLabels(model);

# Once we have the `FaceLabels` object (in this case stored in the variable `labels`), we can add new identifiers (aka "tags") to it. In the next line we create a new tag called `"diri0"` as the union of the objects identified as `"sides"` and `"sides_c"`, which is precisely what we need to represent the Dirichlet boundary $\Gamma_0$.

add_tag_from_tags!(labels,"diri0",["sides", "sides_c"]);

# We follow the same approach to build a new identifier for the Dirichlet boundary $\Gamma_g$. In this case, objects in $\Gamma_g$ can be expressed as the union of the objects identified with the tags `"circle"`, `"circle_c"`, `"triangle"`, `"triangle_c"`, `"square"`, `"square_c"`. Thus, we create a new tag for  $\Gamma_g$, called `"dirig"` simply as follows:


add_tag_from_tags!(labels,"dirig",
  ["circle","circle_c", "triangle", "triangle_c", "square", "square_c"])

# ### FE Spaces
#
# Now, we can build the FE spaces by using the newly defined boundary tags.

order = 1
diritags = ["diri0", "dirig"]
V = CLagrangianFESpace(Float64,model,labels,order,diritags);

# Note that, we pass the `labels` variable (that contains the newly created boundary tags) in the third argument of the `CLagrangianFESpace` constructor. From this FE space, we can define the test and trial FE spaces

g = 1.0
V0 = TestFESpace(V)
Ug = TrialFESpace(V,[0.0,g]);

# Note that we set a value of `0.0` on the boundary `"diri0"` and a value of `g=1.0` on the boundary `"dirig"` when constructing the trial FE space as it required by the problem statement. Note that in this tutorial we are passing values instead of functions in order describe the prescribed Dirichlet data since the Dirichlet conditions are described with constant functions in this example.
#
# We can perform a final check to see if we have properly imposed the Dirichlet boundary conditions. If we interpolate a constant function, namely $w(x) = -1$, in the trial FE space $U_g$, the resulting function $w_h$ has to have value equal to -1 in the nodes that are not on the Dirichlet boundary and fulfill the boundary conditions at the nodes on the Dirichlet boundary. The interpolation is done with the `interpolate` function as follows

w(x) = -1.0
wh = interpolate(Ug,w)

# The computed object `wh` is an instance of `FEFunction`. We can visualize it as we have already in previous tutorials:

trian = Triangulation(model)
writevtk(trian,"wh",cellfields=["wh"=>wh])

# If you open the generated file `wh.vtu` with paraview and chose to color the solid by the field `"wh"` you can confirm that the interpolated function fulfills the Dirichlet boundary conditions as expected (see figure below).
#
# ![](../assets/t0041_p_laplacian/wh.png)
#
# ### Non-linear FE problem
#
#
# At this point, we are ready to define the non-linear FE problem. To this end, we need to define the weak residual and also its corresponding Jacobian. The particular way this is done is similar as the strategy seen in previous tutorials to define the term in the weak form in a linear problem. We will also use types inheriting from the abstract type `FETerm` to define the different terms of the problem.  In this case, instead of an `AffineFETerm` (which is for linear problems), we use a `NonLinearFETerm`. An instance of `NonLinearFETerm` is constructed as follows. First, we need to define the integrand of the weak residual. In this case:

using LinearAlgebra: norm
const p = 3
@law flux(x,∇u) = norm(∇u)^(p-2) * ∇u
f(x) = 1.0
res(u,v) = inner( ∇(v), flux(∇(u)) ) - inner(v,f)

# Function `res` is the one representing the integrand of the weak residual $[r(u)](v)$. The first argument of function `res` represents the function $u\in U_g$ where the residual is evaluated. The second argument represents a generic test function $v\in V_0$. Note that the notation we have used to define this function is the same as the one we have used in previous tutorials for linear problems. In particular, we have used the macro `@law` to construct a constitutive relation (in that case the non-linear flux associated with the gradient of the solution).
#
# On the other hand, we need to define the (integrand of the) Jacobian associated with this residual, which is done as follows


@law dflux(x,∇du,∇u) = (p-2)*norm(∇u)^(p-4)*inner(∇u,∇du)*∇u + norm(∇u)^(p-2) * ∇du
jac(u,v,du) = inner(  ∇(v) , dflux(∇(du),∇(u)) )

# Function `jac` represents the integrand of the Jacobian $[j(u)](v,\delta u)$ previously defined. The first argument of function `jac` stands for function $u\in U_g$ where the Jacobian is evaluated. The second argument is a test function $v\in V_0$, and finally the third argument represents an infinitesimal solution increment $\delta u \in V_0$. Note that we have also used the macro `@law` to define the "linearization" of the flux.
#
# We can finally build the `NonLinearFETerm` as follows.

quad = CellQuadrature(trian,order=2)
t_Ω = NonLinearFETerm(res,jac,trian,quad)

# Note that we pass in the first and second arguments the functions that represent the intgrands of the residual and Jacobian. The other two arguments, are the triangulation and quadrature used to perform the integrals numerically on the corresponding domain (in this case the volume $\Omega$).
#
# From this `NonLinearFETerm` object, we finally construct the non-linear FE problem as follows.

op = NonLinearFEOperator(V,Ug,t_Ω)

# In previous line, we have constructed an instance of `NonLinearFEOperator`, which is the type that represents a general non-linear FE problem in Gridap. The constructor takes the test and trial spaces of the problem, and the `FETerms` objects describing the corresponding weak form.


# ### Non-linear solver phase
#
# We have already built the non-linear FE problem. Now, the remaining step is to solve it. In Gridap, non-linear (and also linear) FE problems can be solved with instances of the type `NonLinearFESolver`. The type `NonLinearFESolver` is a concrete implementation of the abstract type `FESolver` particularly designed for non-linear problems (in contrast to the concrete type `LinearFESolver` which is for the linear case). 
#
# A `NonLinearFESolver` is constructed from an algebraic non-linear solver (e.g., a Newton-Raphson solver, a trust-region solver, etc.). In Gridap non-linear algebraic solvers are represented by types inheriting from the abstract type `NonLinearSolver`. Once of the concrete implementations of this abstract type available in Gridap is the `JuliaNLSolver`, which uses the `nlsove` function of the official Julia package [NLsolve](https://github.com/JuliaNLSolvers/NLsolve.jl) to solve the underlying non-linear algebraic problem.
#
# We construct an instance of `JuliaNLSolver` as follows:

using LineSearches: BackTracking

ls = BackslashSolver()
nls = JuliaNLSolver(
  ls; show_trace=true, method=:newton, linesearch=BackTracking())

# The first argument of the `JuliaNLSolver` constructor takes a single positional argument and several keyword arguments. In the positional argument, we pass the linear solver we want to use at each non-linear iteration (if the chosen non-linear solution method requires to solve linear systems of algebraic equations at each iterations). In this case, we use a `BackslashSolver` which is a wrapper of the Julia built-in "backslash" operator. On the other hand, the valid key word arguments are the same as the ones of function `nlsolve` of the [NLsolve](https://github.com/JuliaNLSolvers/NLsolve.jl) package (see the documentation of this package for more information). Note that we are selecting a Newton-Raphson method with a back-traking line-search function. The other keyword arguments are to show and to store a trace of the iterations.

# Now, we are finally in place to build the `NonLinearFESolver` object:

solver = NonLinearFESolver(nls)


# To finally solve the non-linear FE problem, we need to chose an initial guess. The initial guess is a `FEFunction`, which is in this case is build from a vector for random nodal values:

import Random
Random.seed!(1234)

x = rand(Float64,num_free_dofs(Ug))
uh = FEFunction(Ug,x)

# Using, the initial guess and the non-linear FE solver, we solve the problem as follows:

solve!(uh,solver,op)

#md # If you run previous line in a jupyter notebook, you will see a trace as this one
#md # ```
#md # Iter     f(x) inf-norm    Step 2-norm 
#md # ------   --------------   --------------
#md #      0     1.139082e+01              NaN
#md #      1     2.849303e+00     2.361896e+02
#md #      2     7.176996e-01     6.262418e+01
#md #      3     1.917792e-01     1.761268e+01
#md #      4     5.525576e-02     4.295340e+00
#md #      5     1.186876e-02     6.847898e-01
#md #      6     2.359521e-03     7.063845e-02
#md #      7     3.170074e-04     5.936403e-03
#md #      8     6.754149e-05     5.142141e-04
#md #      9     1.195143e-05     4.066167e-05
#md #     10     2.308345e-06     1.832637e-06
#md #     11     8.679377e-08     2.492892e-08
#md #     12     1.375616e-10     3.492308e-11
#md # ```

# Note that the solve! function updates the given initial guess with the solution of the problem.  That is, once function `solve!` returns, the variable `uh` contains the solution of the problem. To visualize it, execute following line and inspect the generated file with paraview.

writevtk(trian,"results",cellfields=["uh"=>uh])

# ![](../assets/t0041_p_laplacian/sol-plap.png)
#
# Congratulations, another tutorial done!
