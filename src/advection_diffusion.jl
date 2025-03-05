# In this tutorial, we will learn:
#    -  Advection and diffusion term explanation
#    -  How to create models with arbitrary geometry in Gmsh
#    -  How to define the equation in the weak form
#    -  How to solve for and visualize the results

# ## Problem statement

# In this tutorial we will solve the steady-state advection-diffusion equation. 

# In fluid mechanics, advection refers to the transport of a substance or property (such as heat, mass and momentum) by the motion of the fluid itself. It is a key mechanism in which quantities are carried along with the bulk movement of the fluid. As opposed to advection, diffusion refers to the process where substances spread out due to concentration gradients, moving from regions of higher concentration to regions of lower concentration.

# The advection-diffusion equation combines the effects of both mechanisms, making it a powerful tool for modeling real-world transport processes where substances are not only carried by the flow of a fluid but also spread out over time due to diffusion. The best example of an advection-diffusion equation in the real case is the temperature distribution in a flow field.

# Formally, a steady-state advection-diffusion equation with Dirichlet boundary conditions could be defined as:

# $ \nabla \cdot \left ( D\nabla T - \vec{V}T  \right ) =0 $ ,\
# $ T = g $ on boundaries,

# where:
#    -  $T$ is the scalar quantity being transported (here T refers to temperature);
#    -  $\vec{V}$ is the velocity vector of the fluid (representing advection);
#    -  $D$ is the diffusion coefficient (a constant that characterizes the rate of diffusion).

# In order to help users get to know the strong feasibility of Gmsh, where users can create the geometry needed for Gridap, we will define the domain in the shape of a pentagon. We will see both result plottings with and without advection effect.

# ## Model geometry in Gmsh

# This is one of  the most important parts of this tutorial. In this case, we make fully use of `Gmsh`, an open-source finite element mesh generator with a built-in CAD engine and post-processor. Gmsh is the tool highly bound to `Gridap.jl`, and it has its own scripting language (`.geo` scripts), and the scripts could be loaded to a very nice-designed user interface, `ONELAB`. In `ONELAB`, you could easily run the script you wrote, and generate the mesh you want. Here we would illustrate the whole process of creating a 2D model in a pentagon shape using `Gmsh` step by step.

# Note that all the code in this part ("Model geometry in Gmsh") is for the `.geo` scripts.

# First thing we need to create is the point, which is the simplest entity we can define in `Gmsh`. Note that in `Gmsh`, all points are uniquely identified by a tag (a strictly positive integer) and defined by a list of four numbers: three coordinates (`X`, `Y` and `Z`) and the target mesh size (we always use `lc`, which stands for characteristic length) close to the point.

# Here we would like to define a pentagon with side length of 2, and `lc` as following:
# ```js
# lc = 1e-1;
# ```

# So, all five points could be defined as:
# ```js
# Point(1) = {0, -1.0000, 0, lc};
# Point(2) = {0, 1.0000,  0, lc};
# Point(3) = {1.9015, 1.6190, 0, lc};
# Point(4) = {3.0777, 0, 0, lc};
# Point(5) = {1.9015, -1.6190, 0, lc};
# ```

# Once we have the points, we could naturally try to define the lines. Obviously here we have 5 sides of the pentagon to define. The definition of lines is pretty much the same as points. A straight line is defined by a list of two-point tags, and itself can also be identified by a tag.
# ```js
# Line(1) = {1, 2};
# Line(2) = {2, 3};
# Line(3) = {3, 4};
# Line(4) = {4, 5};
# Line(5) = {5, 1};
# ```

# With the lines defined, we can now define the curve loop, which we could utilize to make a surface. A curve loop is defined by an ordered list of connected lines.
# ```js
# Curve Loop(1) = {1, 2, 3, 4, 5};
# ```

# And now is the final step, that is to define the surface. Until now Gmsh has totally understand what we are going to design with the model.
# ```js
# Plane Surface(1) = {1};
# ```

# There is another option here: we can try to define physical groups among the entities we defined before. By defining physical groups, we can better assign the “roles” to the entities we defined. Here for example:
# ```js
# Physical Line("l1") = {1};
# Physical Line("l2") = {2};
# Physical Line("l3") = {3};
# Physical Line("l4") = {4};
# Physical Line("l5") = {5};
# ```

# Once we have the model geometry, we can simply click `2D` under the `mesh` category and then click `save`. Then Gmsh would generate a `.msh` file for us. Or we could use code below to ouput the `.msh` file directly:
# ```js
# Mesh 2;
# Save "pentagon_mesh.msh";
# ```

# The meshed model is like following:

# ![Pentagon_2D_Mesh](Pentagon_2D_Mesh.png)

# ## Numerical Scheme

# The weak form of a PDE is a reformulation that allows the problem to be solved in a broader function space. Instead of requiring the solution to satisfy the PDE at every point (as in the strong form), the weak form requires the solution to satisfy an integral equation. This makes it particularly suitable for numerical methods such as the Finite Element Method.

# Since we already hcave the original PDE (strong form), we can multiply each side by a test function $v \in H^1(\Omega)$(functions with square-integrable first derivatives). $v$ satisfies the Dirichlet boundary condition of $0$. Make integral on both sides.

# The weak form associated with this formulation is, find $T \in H^1(\Omega)$, such that:
# $$
# \int_\Omega v (\mathbf{u} \cdot \nabla T) \, \mathrm{d}\Omega 
# + \int_\Omega D \nabla T \cdot \nabla v \, \mathrm{d}\Omega 
# = 0
# $$

# ## FE spaces         

# First, import `Gridap.jl`.

using Gridap
using GridapGmsh

# Import the model from file using GmshDiscreteModel:

model_pentagon = GmshDiscreteModel("pentagon_mesh.msh")

# Set up the test FE space $V_0$, which conforms the zero value boundary conditions.

order = 1
reffe = ReferenceFE(lagrangian,Float64,order)
V_0 = TestFESpace(model_pentagon,reffe;conformity=:H1,dirichlet_tags=["l1","l2","l3","l4","l5"])

# Set up the boundary conditions. Here we set 5 values, representing 5 different temperatures on 5 sides of the pentagon. Notw that $l1$ is the most left vertical side, we would like to define this as 200. We define the middle two 2 sides as 100, and the most right two sides as 0. Define the trial space $U_g$.

boundary_cond = [200,100,0,0,100];
U_g = TrialFESpace(V_0,boundary_cond)

# Do the integration over domain $\Omega$.

degree = 2
Ω = Triangulation(model_pentagon)
dΩ = Measure(Ω,degree)

# ## Weak form

# Define the velocity field $\vec{V}T = (u_1,u_2)$. Here we have two options: 
#    -  zero velocity;
#    -  $u_1 = 0$, and $u_2 = 2$. That means the flow is going up at a velocity of 2.

velocity_zero = VectorValue(0.0, 0.0);

velocity_nonzero = VectorValue(0.0, 2.0);

# Duffusion coefficient is defined as:

D = 0.1;

# The weak form can thus be represented as:

a_zero(u, v)     = ∫(v * (velocity_zero ⋅∇(u))    + ∇(v) ⋅ (D * ∇(u))) * dΩ
a_non_zero(u, v) = ∫(v * (velocity_nonzero ⋅∇(u)) + ∇(v) ⋅ (D * ∇(u))) * dΩ
b(v) = 0.0

# ## Solution

# Now build the FE problem and use the solver. We are solving for both zero and non-zero flow velocity.

op_zero = AffineFEOperator(a_zero,b,U_g,V_0)
uh_zero = Gridap.Algebra.solve(op_zero)

op_non_zero = AffineFEOperator(a_non_zero,b,U_g,V_0)
uh_non_zero = Gridap.Algebra.solve(op_non_zero)

# Ouput the result as a `.vtk` file.

writevtk(Ω,"results_zero",cellfields=["uh_zero"=>uh_zero])

writevtk(Ω,"results_non_zero",cellfields=["uh_non_zero"=>uh_non_zero])

# ## Visualization

# We can use the ParaView to preview the results clearly. Here is the temperature distribution without any flow velocity:

# ![Result_zero](Result_zero.png)

# Here is the temperature distribution with a flow velocity of 2 going up:

# ![Result_zero](Result_non_zero.png)
