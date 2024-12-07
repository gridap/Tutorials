# # Tutorial: `Gridap.jl`  workflow, using the Laplace Equation as an example

# This is an introduction tutorial for `Gridap.jl`, but also a short recap of some classic PDEs in real applications, especially in mechanical engineering. If you are a student just entered the college in mechanical engineering related majors, or you just want to take a glance at how PDE works in real cases, then you might want to take a look at this tutorial.

# The most classical way to solve certain categories of PDEs is separation of variables, and probably representing the results in Fourier’s Series. However, those results are so analytical, you even have no idea whether you results makes sense or not, since it is really hard to visualize. And you might also have learned some numerical techniques like finite difference in dealing with steady-state temperature distribution problems or finite element methods in dealing with elasticity problems. That is a huge system of linear equations, and once you want to play something more complicated, you will feel hard with generating those matrices by yourself.

# This tutorial aims to give you an open option to try PDEs in literally any domain shape using Finite Element Methods. You will not only use `Gridap.jl`, but start from creating model geometry and mesh in Gmsh, then converting the model into a `Gridap.jl` compatible style, applying boundary conditions, solving the system, and visualizing your results. The key point here is that you will never need to worry about setting up the complicated linear systems by yourself, all you need to pay attention is the equation itself, the domain, and boundary conditions.

# **In this tutorial, we will learn:**
# - Basic knowledge for partial differential equations
# - Diffusion term explanation
# - How to create a model using Gmsh, and load the model in Gridap.jl
# - Solve equation and visualize the results

# ## Back to what we might already know about PDE

# Partial Differential Equations (PDEs) are mathematical tools essential for modeling systems that change continuously across space and time. In engineering, they describe phenomena like fluid flow (e.g., Navier-Stokes equations), heat transfer, and material stress, enabling the design of efficient systems such as turbines, thermal devices, and structures. PDEs are also critical in fields like acoustics, electromagnetics, and biomedical engineering, providing solutions to complex real-world problems.

# **We all know some very classic forms of PDE:**
# - The Laplace equation, which is always used to describe the equilibrium temperature distribution of a homogeneous solid;
# - The Poisson Equation, is a modified version of the Laplace Equation, where we introduced the source term;
# - Heat Equation, which describes the time-dependent distribution of temperature in a material due to heat conduction.
# - Navier-Stocks, a very classic non-linear function, which includes time derivative, diffusion, advection, and pressure gradient. It is the Newton’s Second Law in the fluid world, and we are always trying to solve this function under certain assumptions.

# ## About fluid dynamics, diffusion terms and advection terms

# In fluid mechanics, advection refers to the transport of a substance or property (such as heat, mass and momentum) by the motion of the fluid itself. It is a key mechanism in which quantities are carried along with the bulk movement of the fluid. 

# As opposed to processes like advection, diffusion refers to the process where substances spread out due to concentration gradients, moving from regions of higher concentration to regions of lower concentration. It occurs at the molecular level and is typically much slower than advection. Diffusion is modeled by Fick's Law and is a passive process that does not rely on bulk fluid movement.

# The advection-diffusion equation combines the effects of both mechanisms, making it a powerful tool for modeling real-world transport processes where substances are not only carried by the flow of a fluid but also spread out over time due to diffusion. It can be applied to describe multiple real scientific problems in fields like environmental engineering (modeling the dispersion of pollutants in the atmosphere), heat transfer (Describing the combined effects of convective heat transfer and thermal conduction) and even chemical engineering (understanding the transport of reactants and products in chemical reactors).

# In this tutorial, we will try to simulate the process of diffusion in a domain of a pentagon, which is the same process as temperature distribution. 

# ## Creating model geometry and mesh

# This would be the most important part of this tutorial since we are already pretty familiar with classic terms in PDEs, but we are always facing difficulties defining our domains in traditional programming environments. It is relatively easy to reshape a triangle domain into a column vector, but for domains in other shapes, it might be difficult. And real case, say for a temperature field, the domain is not always in a squared shape.

# In this case, we make fully use of `Gmsh`, an open-source 3D finite element mesh generator with a built-in CAD engine and post-processor. Gmsh is the tool highly bounded to `Gridap.jl`, and it has its own scripting language (`.geo` scripts), and the scripts could be loaded to a very nice-designed user interface, `ONELAB`. In `ONELAB`, you could easy run the script you wrote, and generate the mesh you want. Here we would illustrate the whole process of creating a 2D model in a pentagon shape with a using `Gmsh` step by step.

# First, we need to create points, which is the simplest entity we can define in `Gmsh`. Note that in `Gmsh`, all points are uniquely identified by a tag (a strictly positive integer) and defined by a list of four numbers: three coordinates (`X`, `Y` and `Z`) and the target mesh size (we always use `lc`, which stands for characteristic length) close to the point. 

# Here we would like to define a pentagon with side length of 1, and `lc` as following:
# ```js
# lc = 1e-1;
# ```

# So, all five points could be defined as:
# ```js
# Point(1) = {.8507, 0, 0, lc};
# Point(2) = {0.2629, 0.8090,  0, lc};
# Point(3) = {-0.6882, 0.5000, 0, lc};
# Point(4) = {-0.6882, -0.5000, 0, lc};
# Point(5) = {0.2629, -0.8090, 0, lc};
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

# Now, The model can be imported using GmshDiscreteModel:

using Gridap
using GridapGmsh

model_pentagon = GmshDiscreteModel("pentagon_mesh.msh ")


# We finally got the mesh like following in Gmsh user interface:

# ![Mesh](../assets/gmsh_gridap/Mesh.png)

# ## About the solver

# Now it is the time to set up the boundary conditions. Here we set 5 values, representing 5 different temperatures on 5 sides of the pentagon.

boundary_cond = [500,400,300,200,100];


# Now we need to setup the Finite Element Space.

order = 1
reffe = ReferenceFE(lagrangian,Float64,order)
V0 = TestFESpace(model123,reffe;conformity=:H1,dirichlet_tags=["l1","l2","l3","l4","l5"])

Ug = TrialFESpace(V0,boundary_cond)


# Now we are going to read the domain and do the triangulation:

degree = 2
Ω = Triangulation(model123)
dΩ = Measure(Ω,degree)


# And we will define the weak form of the Laplace Equation:

a(u,v) = ∫( ∇(v)⋅∇(u) )*dΩ
b(v) = ∫( 0*v )*dΩ


# And we can finally apply the solver:

op = AffineFEOperator(a,b,Ug,V0)
uh = Gridap.Algebra.solve(op)


# ## Visualization

# Since we already got the result `uh`, we write the result as a `.vtk` file:

writevtk(Ω,"pentagon_result",cellfields=["uh"=>uh])


# We can use the ParaView to preview the results clearly.

![ParaView_Result](../assets/gmsh_gridap/ParaView_Result.png)
