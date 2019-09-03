# # Tutorial 3: Linear elasticity
#
#md # [![](https://mybinder.org/badge_logo.svg)](@__BINDER_ROOT_URL__/notebooks/t003_elasticity.ipynb)
#md # [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/notebooks/t003_elasticity.ipynb)
# 
# ## Learning outcomes
#
# - How to approximate vector-valued problems
# - How to solve problems with complex constitutive laws
# - How to impose Dirichlet boundary conditions only in selected components
# - How to impose Dirichlet boundary conditions described by more than one function
# - How to deal with multi-material problems
#
# ## Problem statement
# 
# In this tutorial, we detail how to solve a linear elasticity problem. We consider the geometry depicted in the next figure.
#
# ![](../models/solid.png)
#
# The mechanical problem is defined with the following boundary conditions. All components of the displacement vector are constrained to zero on the surface $\Gamma_{\rm G}$, which is marked in green in the figure. On the other hand, the first component of the displacement vector is prescribed to the value $\delta\doteq 5$mm on the surface $\Gamma_{\rm B}$, which is marked in blue. No body or surface forces are included in this example.
#
#
# The PDE to solve is
#
# ```math
# \left\lbrace
# \begin{aligned}
# -∇\cdot\sigma(u) = 0 \ \text{in} \ \Omega\\
# u = 0 \ \text{on}\ \Gamma_{\rm G}\\
# u_1 = \delta \ \text{on}\ \Gamma_{\rm B}\\
# \nabla u\cdot n = 0 \ \text{on}\  \Gamma_{\rm N}\\
# \end{aligned}
# \right.
# ```
#
# The variable `u` stands for the unknown displacement vector. The vector $n$ is the unit outward normal to the Neumann boundary $\Gamma_{\rm N}\doteq\partial\Omega\setminus\left(\Gamma_{\rm B}\cup\Gamma_{\rm G}\right)$ and $\sigma(u)$ is the stress tensor defined as
#
# ```math
# \sigma(u) \doteq \lambda\ {\rm tr}(\varepsilon(u)) \ I +2 \mu \  \varepsilon(u),
# ```
#
# where $I$ is the 2nd order identity tensor, $\lambda$ and $\mu$ are the *Lamé parameters* of the material, and $\varepsilon(u)\doteq\frac{1}{2}\left(\nabla u + (\nabla u)^t \right)$ is the symmetric gradient (i.e., the strain tensor).
#
# ## Numerical scheme
#
# In this tutorial, we use a conventional Galerkin FE method with conforming Lagrangian FE spaces. For this formulation, the weak version of the problem is find $u\in U$ such that $ a(v,u) = 0 $ for all $v\in V_0$, where $U$ is the subset of functions in $V\doteq[H^1(\Omega)]^3$ that fulfill the Dirichlet boundary conditions of the problem, whereas $V_0$ are functions in $V$ fulfilling $v=0$ on $\Gamma_{\rm G}$ and $v_1=0$ on $\Gamma_{\rm B}$. The bilinear form of the problem is
#
# ```math
# a(v,u)\doteq \int_{\Omega} \varepsilon(v) : \sigma(u) \ {\rm d}\Omega.
# ```
# ## Implementation
#
# ### Discrete model
#
# We start by loading the discrete model from a file

using Gridap

model = DiscreteModelFromFile("../models/solid.json");

# Write the model to vtk with the command

writevtk(model,"model");

# and open the resulting files with Paravaiew in order to inspect it. As you will see, the discretization is done with linear tetrahedral elements. Note also that the boundaries $\Gamma_{\rm B}$ and $\Gamma_{\rm G}$ are identified in the model with the names `"surface_1"` and `"surface_2"` respectively.  For instance, if you visualize the faces of the model and color them by the field `"surface_2"`, you will see that only the faces on $\Gamma_{\rm G}$ have a value different from zero.
#
# ![](../models/solid-surf2.png)
#
# ### Vector-valued FE space
#
# The next step is the construction of the FE space. The main difference with respect to the previous tutorials that discussed the Poisson problem is that we need a vector-valued FE space to solve the current problem. This is achieved as follows

const T = VectorValue{3,Float64}
diritags = ["surface_1","surface_2"]
dirimasks = [(true,false,false), (true,true,true)]
order = 1
V = CLagrangianFESpace(T,model,order,diritags,dirimasks);

# In the construction of the vector-valued FE space, there are two new concepts deserving some discussion. Note that, in the first argument of the constructor `CLagrangianFESpace`, we pass the type `Vectorvalue{3,Float64}`, which is the way Gridap represents vectors of three `Float64` components.  Another major difference with respect to previous tutorials is the presence of the argument `dirimasks`. This argument allows one to chose which components of the displacement are constrained on the Dirichlet boundary and which are not. Note that we constrain only the first component on the boundary $\Gamma_{\rm B}$ (i.e., `"surface_1"`), whereas we constrain all components on $\Gamma_{\rm G}$ (i.e., `"surface_2"`) as it defined in the problem statement.
#
# At this point, we can define the test and trial spaces. The test space is built as we have detailed in previous tutorials

V0 = TestFESpace(V);

# However, the construction of the trial space is slightly different in this case. The Dirichlet boundary conditions are described with two different functions, one for boundary $\Gamma_{\rm B}$ and another one for $\Gamma_{\rm G}$. These functions can be defined as

g1(x) = VectorValue(0.005,0.0,0.0)
g2(x) = zero(T)

# Note that it is irrelevant which values we use in the second and third components of the vector returned by function `g1` since only the first component is constrained on the boundary $\Gamma_{\rm B}$ (i.e., the two last components will be ignored by the code).  Note also that we have used the function `zero` in function `g2` to construct the zero vector of three components.

#
# From functions `g1` and `g2`, we define the trial space as follows:

U = TrialFESpace(V,[g1,g2]);

# Notet that the functions `g1` and `g2` are passed as a vector of functions to the `TrialFESpace` constructor, one function for each boundary identifier passed previously in the `diritags` argument of the `CLagrangianFESpace` constructor.


# ### Constitutive law
#
# In this example, the definition of the terms  in the weak form requires more work that in previous tutorials, in particular, for the definition of the bilinear form.  In this case, the integrand of the bilinear form is defined as
#

a(v,u) = inner( ε(v), σ(ε(u)) )

# Note that we have used the Gridap function `ε` (aka `symmetric_gradient`) to compute the symmetric gradient of the test and trial functions. On the other hand, `σ` is a function (to be defined below) that computes the stress tensor associated with `ε(u)`. In Gridap, function `σ` and other types of constitutive laws are defined by using the supplied macro `@law`:

const E = 70.0e9
const ν = 0.33

const λ = (E*ν)/((1+ν)*(1-2*ν))
const μ = E/(2*(1+ν))

@law σ(x,ε) = λ*tr(ε)*one(ε) + 2*μ*ε

# The macro `@law` is always placed before a function definition.  The arguments of the function annotated with the `@law` macro represent the values of different quantities at a generic integration point. The first argument always represents the coordinate of the integration point. The rest of the argument have arbitrary meaning. In this example, the second argument represents the strain tensor at an integration point, from which the stress tensor is computed. Note that we have used the function `tr` and function `one` to compute the stress tensor. We have used material parameters corresponding to aluminum.
#
# The macro `@law` automatically adds an extra method to the corresponding function. The generated method has always an argument less that the original function definition (i.e., the first argument is removed). You can easily check it with following line

methods(σ)

#md # If you run previous in a notebook, we will see
#nb # Note
# that function `σ` has indeed two methods: the one we have defined, which has 2 arguments, and another one with only a single argument that has been created by the macro. The new method can be used as `σ(ε(u))` where `u` is a trial function in the definition of a bilinear form (as done above), or where `u` is a `FEFunction` (we will use this for writing the stress tensor into the vtk file).
#
#
# ### Solution of the FE problem
#
# The remaining steps for solving the FE problem are as in previous tutorials. We build the integration mesh and quadrature for integrating in the volume

trian = Triangulation(model)
quad = CellQuadrature(trian,order=2);

# We define the FE problem

t_Ω = LinearFETerm(a,trian,quad)
op = LinearFEOperator(V0,U,t_Ω);

# and we solve it

uh = solve(op);

# Note that in the construction of the `LinearFEOperator` we have used a `LinearFETerm` instead of an `AffineFETerm` as it was done in previous tutorial. The `LinearFETerm` is a particular implementation of `FETerm` for terms that only contribute to the system matrix (and not to the right hand side vector). This is what we want in this example since the body forces are zero.
#
# Finally, we write the results to a file

writevtk(trian,"results",cellfields=["uh"=>uh,"epsi"=>ε(uh),"sigma"=>σ(ε(uh))])

# Note that we are also including the strain and stress tensors into the results file. Open it, to see the computed solution of the problem. You can clearly see that the surface  $\Gamma_{\rm B}$ is pulled in $x_1$-direction and that the solid deforms accordingly (in the figure below deformation magnified 40 times).
#
# ![](../assets/t003_elasticity/disp_ux_40.png)


# ## Multi-material problems
#
# In practical applications it is often the case that a component is made of several materials with different material properties, and therefore, the underlying FE code has to be able to deal with this. We end this tutorial by extending previous code to deal with multi-material problems. Let us consider that the piece simulated before is now made of 2 different materials (see next figure). In particular, we assume that the volume depicted in dark green is made of aluminum, whereas the volume marked in purple is made of steel.
#
# ![](../models/solid-mat.png)
#
# The first thing we need is that the two different material volumes are properly identified in the model. To check this, inspect the model with Paraview (by writing it to vtk format as done before). Note that the volume made of aluminum is identified as `"material_1"`, whereas the volume made of steel is identified as `"material_2"`.
#
# The following two lines build a vector, namely `tags`, whose length is the number of cells in the model and for each cell contains an integer that identifies the material of the cell. 

labels = FaceLabels(model)
dimension = 3
tags = first_tag_on_face(labels,dimension);

# We use the following line in order to retrieve the integer value associated with `"material_1"`.

const alu_tag = tag_from_name(labels,"material_1")

# That is, all cells whose corresponding value in the `tags` vector is `alu_tag` are made of aluminum, otherwise they are made of steel since there are only two materials in this example.
#
# At this point, we are ready to define the multi-material constitutive law. First, we define the material parameters for aluminum and steel respectively:

function lame_parameters(E,ν)
  λ = (E*ν)/((1+ν)*(1-2*ν))
  μ = E/(2*(1+ν))
  (λ, μ)
end

const E_alu = 70.0e9
const ν_alu = 0.33
const (λ_alu,μ_alu) = lame_parameters(E_alu,ν_alu)

const E_steel = 200.0e9
const ν_steel = 0.33
const (λ_steel,μ_steel) = lame_parameters(E_steel,ν_steel)

# Then, we define the function representing the constitutive law

@law function σ_bimat(x,ε,tag)
  if tag == alu_tag
    return λ_alu*tr(ε)*one(ε) + 2*μ_alu*ε
  else
    return λ_steel*tr(ε)*one(ε) + 2*μ_steel*ε
  end
end

# Note that the last argument represents the integer value associated with a certain material. If the value corresponds to the one for aluminum, then we use the constitutive law for this material, otherwise, we use the law for steel. With the generated constitutive law, we can re-define the bilinear form of the problems:

a(v,u) = inner( ε(v), σ_bimat(ε(u),tags) )

# Note that we have passed the vector containing the material tags in the last argument of the function `σ_bimat`.
#
# At this point, we can build the FE problem again and solve it

t_Ω = LinearFETerm(a,trian,quad)
op = LinearFEOperator(V0,U,t_Ω)
uh = solve(op);

# One the solution is computed, we can store the results in a file for visualization. Note that, we are also including the stress tensor in the file (computed with the bi-material law).

writevtk(trian,"results",cellfields=
  ["uh"=>uh,"epsi"=>ε(uh),"sigma"=>σ_bimat(ε(uh),tags)])


#  Tutorial done!
