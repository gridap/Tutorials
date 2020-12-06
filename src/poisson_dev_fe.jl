# --- TO RE-THINK ---
# Disclaimer: This tutorial is about a low-level definition of Finite Element (FE)
# methods. It would be convenient to have two (or three) previous tutorials, about
# the three data type hierarchies rooted at `Map`, `Field`, and `CellDatum`,
# the details underlying the `lazy_map` generic function, and the related
# `LazyArray`, and the combination of these to create lazy mathematical expressions
# that involve fields. This is work in progress.
# --- TO RE-THINK ---

# This tutorial is advanced and you only need to go through this if you want
# to know the internals of `Gridap` and what it does under the hood.
# Even though you will likely want to use the high-level APIs in `Gridap`,
# this tutorial will (hopefully) help if you want to become a `Gridap` developer,
# not just a user. We also consider that this tutorial shows how powerful and
# expressive the `Gridap` kernel is, and how mastering it you can implement new
# algorithms not currently provided by the library.

# It is highly recommended (if not essential) that the tutorial is followed together with a Julia
# debugger, e.g., the one which comes with the Visual Studio Code (VSCode) extension
# for the Julia programming language. Some of the observations that come along with
# the code snippet are quite subtle/technical and may require a deeper exploration
# of the underlying code using a debugger.

# Let us start including `Gridap` and some of its submodules, to have access to
# a rich set of not so high-level methods. Note that the module `Gridap` provides
# the high-level API, whereas the submodules like `Gridap.FESpaces` provide access to
# the different parts of the low-level API.

using Gridap
using Gridap.FESpaces
using Gridap.ReferenceFEs
using Gridap.Arrays
using Gridap.Geometry
using Gridap.Fields
using Gridap.CellData
using FillArrays
using Test

# We first create the geometry model and FE spaces using the high-level API.
# In this tutorial, we are not going to describe the geometrical machinery in detail,
# only what is relevant for the discussion. To simplify the analysis of the outputs,
# you can consider a 2D mesh, i.e., `D=2` (everything below works for any spatial dimension
# without any extra complication). In order to make things slightly more interesting,
# i.e., having non-constant Jacobians, we have considered a mesh that is
# a stretching of an equal-sized structured mesh.

L = 2 # Domain length in each space dimension
D = 2 # Number of spatial dimensions
n = 4 # Partition (i.e., number of cells per space dimension)

function stretching(x::Point)
   m = zeros(length(x))
   m[1] = x[1]^2
   for i in 2:D
     m[i] = x[i]
   end
   Point(m)
end

pmin = Point(Fill(0,D))
pmax = Point(Fill(L,D))
partition = Tuple(Fill(n,D))
model = CartesianDiscreteModel(pmin,pmax,partition,map=stretching)

# The next step is to build the global FE space of functions from which we are
# going to extract the unknown function of the differential problem at hand. This
# tutorial explores the Galerkin discretization of the scalar Poisson equation.
# Thus, we need to build H1-conforming global FE spaces. This can be achieved using $C^0$
# continuous functions made of piece(cell)-wise polynomials. This is precisely the purpose
# of the following lines of code.

# First, we build a scalar-valued (`T = Float64`) Lagrangian reference FE of order `order`
# atop a reference n-cube of dimension `D`. To this end, we first need to create a `Polytope`
# using an array of dimension `D` with the parameter `HEX_AXIS`, which encodes the reference
# representation of the cells in the mesh. Then, we create the Lagrangian reference FE using the
# reference geometry just created in the previous step. It is not the purpose of this tutorial to
# describe the (key) abstract concept of `ReferenceFE` in Gridap.

T = Float64
order = 1
pol = Polytope(Fill(HEX_AXIS,D)...)
reffe = LagrangianRefFE(T,pol,order)

# Second, we build the test (Vₕ) and trial (Uₕ) global finite element (FE) spaces
# out of `model` and `reffe`. At this point we also specify the notion of conformity
# that we are willing to satisfy, i.e., H1-conformity, and the region of the domain
# in which we want to (strongly) impose Dirichlet boundary conditions, the whole
# boundary of the box in this case.

Vₕ = FESpace(model,reffe;conformity=:H1,dirichlet_tags="boundary")

u(x) = x[1]            # Analytical solution (for Dirichlet data)
Uₕ = TrialFESpace(Vₕ,u)

# We also want to extract the triangulation of the model and obtain a quadrature.
Tₕ = Triangulation(model)
Qₕ = CellQuadrature(Tₕ,2*order)

# Qₕ is an instance of type `CellQuadrature`, a subtype of the `CellDatum` abstract
# data type.

isa(Qₕ,CellDatum)
subtypes(CellDatum)

# `CellDatum` is the root of one out of three main type hierarchies in Gridap
# (along with the ones rooted at the abstract types `Map` and `Field`) on which the
# the evaluation of variational methods in finite-dimensional spaces is grounded on.
# Any developer of Gridap should familiarize with these three hierachies to some extent.
# Along this tutorial we will give some insight on the rationale underlying these, with
# some examples, but more effort in the form of self-research is expected
# from the reader as well.

# Conceptually, an instance of a `CellDatum` represents a collection of quantities
# (e.g., points in a reference system, or scalar-, vector- or tensor-valued
# fields, or arrays made of these), once per each cell of a triangulation. Using the
# `get_cell_data` generic function one can extract an array with such quantities. For example, in
# the case of Qₕ, we get an array of quadrature rules for numerical integration.

Qₕ_cell_data = get_cell_data(Qₕ)
@test length(Qₕ_cell_data) == num_cells(Tₕ)

# In this case we get the same quadrature rule in all cells (note that the returned array is of
# type `Fill`). Gridap also supports different quadrature rules to be used in different
# cells. Exploring such feature is out of scope of the present tutorial.

# Any `CellDatum` has a trait, the so-called `DomainStyle` trait. This information
# is consumed by `Gridap` in different parts of the code. It specifies whether
# the quantities in it are either expressed in the reference (`ReferenceDomain`) or the physical
# (`PhysicalDomain`) domain. We can indeed check the `DomainStyle` of a `CellDatum` using the
# `DomainStyle` generic function:

Gridap.FESpaces.DomainStyle(Qₕ) == Gridap.FESpaces.ReferenceDomain()
Gridap.FESpaces.DomainStyle(Qₕ) == Gridap.FESpaces.PhysicalDomain()

# If we evaluate the two expressions above, we can see that the `DomainStyle` trait of Qₕ is
# `ReferenceDomain`. This means that the evaluation points of the quadrature rules within Qₕ
# are expressed in the parametric space of the reference domain of the cells. We note that, while
# finite elements may not be defined in this parametric space (it is though standard practice
# with Lagrangian FEs, and other FEs, because of performance reasons), finite element functions are
# always integrated in such a parametric space.

# Using the array of quadrature rules `Qₕ_cell_data`, we can access to any of its entries.
# The object retrieved provides an array of points (`Point` data type in `Gridap`) in the
# cell's reference parametric space $[0,1]^d$ and their corresponding weights.

q = Qₕ_cell_data[rand(1:num_cells(Tₕ))]
p = get_coordinates(q)
w = get_weights(q)

# However, there is a more convenient way (for reasons made clear along the tutorial) to work
# with the evaluation points of quadratules rules in `Gridap`. Namely, using the `get_cell_points` # function we can extract a `CellPoint` object out of a `CellQuadrature`.

Qₕ_cell_point = get_cell_points(Qₕ)

# `CellPoint` (just as `CellQuadrature`) is a subtype of `CellDatum` as well

@test isa(Qₕ_cell_point, CellDatum)

# and thus we can ask for the value of its `DomainStyle` trait, and get an array of quantities out
# of it using the `get_cell_data` generic function

@test Gridap.FESpaces.DomainStyle(Qₕ_cell_point) == Gridap.FESpaces.ReferenceDomain()
qₖ = get_cell_data(Qₕ_cell_point)

# Not surprisingly, the `DomainStyle` trait of the `CellPoint` object is `ReferenceDomain`, and we # get a (cell) array with an array of `Point`s per each cell out of a `CellPoint`. As seen in
# the sequel, `CellPoint`s are relevant objects because they are the ones that one can use
# in order to evaluate the so-called `CellField` objects on the set of points of a `CellPoint`.

# `CellField` is an abstract type rooted at a hierarchy that plays a cornerstone role in the
# implementation of the finite element method in `Gridap`. At this point, the reader should keep
# in mind that the finite element method works with global spaces of functions which are defined
# piece-wise on each cell of the triangulation. In a nutshell (more in the sections
# below), a `CellField`, as it being a subtype of `CellDatum`, might be understood as a
# collection of `Field`s (or arrays made out them) per each triangulation cell. Unlike a plain
# array of `Field`s, a `CellField` is associated to triangulation, and it holds the required
# metadata in order to perform, e.g., the transformations among parametric spaces when taking
# a differential operator out of it (e.g., the pull back of the gradients). For example, a global
# finite element function, or the collection of shape basis functions in the local FE space of
# each cell are examples of `CellField` objects.

## Exploring our first `CellField` objects and its evaluation

# Let us work with our first `CellField` objects. In particular, let us extract out of the global
# test space, Vₕ, and trial space, Uₕ, a collection of local test and trial finite element
# shape basis functions, respectively.

dv = get_cell_shapefuns(Vₕ)
du = get_cell_shapefuns_trial(Uₕ)

# The objects returned are of `FEBasis` type, one of the subtypes of `CellField`.
# Apart from `DomainStyle`, `FEBasis` objects also have an additional trait, `BasisStyle`,
# which specifies wether the cell-local shape basis functions are either of test or
# trial type (in the Galerkin method). This information is consumed in different parts of
# the code.

@test Gridap.FESpaces.BasisStyle(dv) == Gridap.FESpaces.TestBasis()
@test Gridap.FESpaces.BasisStyle(du) == Gridap.FESpaces.TrialBasis()

# As expected, `dv` is made out of test shape functions, and `du`, of trial shape functions.
# We can also confirm that both `dv` and `du` are `CellField` and `CellDatum` objects (i.e.,
# recall that `FEBasis` is a subtype of `CellField`, and the latter is a subtype of
# `CellDatum`).

@test isa(dv,CellField) && isa(dv,CellDatum)
@test isa(du,CellField) && isa(du,CellDatum)

# Thus, one may check the value of their `DomainStyle` trait.

@test Gridap.FESpaces.DomainStyle(dv) == Gridap.FESpaces.ReferenceDomain()
@test Gridap.FESpaces.DomainStyle(du) == Gridap.FESpaces.ReferenceDomain()

# We can see that the `DomainStyle` of both `FEBasis` objects is `ReferenceDomain`.
# In the case of `CellField` objects, this specifies that the point coordinates on which
# we evaluate the cell-local shape basis functions should be provided in the parametric
# space of the reference cell. However, the output from evaluation, as usual in finite elements
# defined parametrically, is the cell-local shape function in the physical domain evaluated at
# the corresponding mapped point.

# Recall from above that `CellField` objects are designed to be evaluated at `CellPoint`
# objects, and that we extracted a `CellPoint` object, `Qₕ_cell_point`, out of a `CellQuadrature`,
# of `ReferenceDomain` trait `DomainStyle`. [SHOULD WE MENTION HERE ANYTHING RELATED TO THE
# `change_domain` FEATURE AND WHY IT CAN BE USEFUL?] Thus, we can evaluate `dv` and `du` at the
# quadrature rule evaluation points, on all cells, straight away as:

dv_at_Qₕ = evaluate(dv,Qₕ_cell_point)
du_at_Qₕ = evaluate(du,Qₕ_cell_point)

# There are a pair of worth noting observations on the result of the previous two instructions.
# First, both `dv_at_Qₕ` and `du_at_Qₕ` are arrays of type `Fill`, thus they provide the same entry
# for whatever index we provide.

dv_at_Qₕ[rand(1:num_cells(Tₕ))]
du_at_Qₕ[rand(1:num_cells(Tₕ))]

# This (same entry) is justified by: (1) the local shape functions
# are evaluated at the same set of points in the reference cell parametric space for
# all cells (i.e., the quadrature rule points), and (2) the
# shape functions in physical space have the same values in all cells at the corresponding mapped
# points in physical space. At this point, the reader may want to observe which objects result from
# the evaluation of, e.g., `dv_at_Qₕ`, at a different set points for each cell (e.g. by building
# its own array of arrays of `Points`).

# Going back to our example, any entry of `dv_at_Qₕ` is a rank-2 array of size 4x4 that provides in # position `[i,j]` the i-th test shape function at the j-th quadrature rule evaluation point.
# On the other hand, any entry of `du_at_Qₕ` is a rank-3 array of size `4x1x4` that provides in
# position `[i,1,j]` the i-th trial shape function at the j-th quadrature point. The reader might
# be wondering why the rank of these two arrays are different. The rationale is that, by means of
# a broadcasted `*` operation of these two arrays, we can get a 4x4x4 array where the `[i,j,k]`
# entry stores the product of the i-th test and j-th trial functions, both evaluated at the k-th
# quadrature point. If we sum over the $k$-index, we get the usual cell-local matrix that
# we assemble into the global matrix in order to get a mass matrix (neglecting the strong
# imposition of Dirichlet boundary conditions). For those readers more used to traditional
# finite element codes, the broadcast followed by the sum over k, is equivalent
# to the following triple for-nested loop:

#   M[:,:]=0.0
#   Loop over quadrature points k
#     Loop over shape test functions i
#       Loop over shape trial functions j
#            M[i,j]+=shape_test[i,k]*shape_trial[i,k]

# Using Julia built-in support for broadcasting, we can vectorize the full operation, and get much
# higher performance.

# The highest-level possible way of performing the aforementioned broadcasted `*` is by building
# a "new" `CellField` instance by multiplying the two `FEBasis` objects, and then evaluating the
# resulting object at the points in `Qₕ_cell_point`. This is something common in `Gridap`. One can
# create new `CellField` objects out of exsting ones, e.g., by performing operations among them, or
# by applying a differential operator, such as the gradient.

dv_mult_du = du*dv
dv_mult_du_at_Qₕ = evaluate(dv_mult_du,Qₕ_cell_point)

# We can check that any entry of the resulting `Fill` array is the `4x4x4` array resulting from the
# broadcasted `*` of the two aforementioned arrays. In order to do so, we can use the so-called
# `Broadcasting(*)` `Gridap` `Map`. This `Map`, when applied to arrays of numbers, essentially
# translates into the built-in Julia broadcast (check that below!). However, as we will see along
# the tutorial, such a `Map` can also be applied to, e.g., (cell) arrays of `Field`s (arrays of
# `Field`s, resp.) to build new (cell) arrays of `Fields` (arrays of `Field`s, resp.). This becomes
# extremely useful to build and evaluate discrete variational forms.

m=Broadcasting(*)
A=evaluate(m,dv_at_Qₕ[rand(1:num_cells(Tₕ))],du_at_Qₕ[rand(1:num_cells(Tₕ))])
B=broadcast(*,dv_at_Qₕ[rand(1:num_cells(Tₕ))],du_at_Qₕ[rand(1:num_cells(Tₕ))])
@test any(A .≈ B)
@test any(A .≈ dv_mult_du_at_Qₕ[rand(1:num_cells(Tₕ))])

# ## Exploring FE functions in `Gridap`

# A FE function with [rand](https://docs.julialang.org/en/v1/stdlib/Random/#Base.rand)
# free dofs in Uₕ can be defined as follows

uₕ = FEFunction(Uₕ,rand(num_free_dofs(Uₕ)))

# We can extract the array that at each cell of the mesh returns a `Field`,
# the FE function restricted to that cell

uₖ = get_cell_data(uₕ)

# An essential part of `Gridap` is the concept of `Field`. It provides an
# abstract representation of a physical field of scalar (e.g., a `Float64`),
# vector (`VectorValue`), or tensor (`TensorValue`) type that takes points
# (`Point`) in the domain in which it is defined and returns the scalar,
# vector, and tensor values, resp. at these points. The method
# `evaluate_field!` implements this operation; it evaluates the field in
# a _vector_ of points (for performance, it is _vectorised_ wrt points). It has
# been implemented with caches for high performance. You can take a look at the
# `Field` abstract type in Gridap at this point, and check its API.

# We can also extract the global indices of the DOFs in each cell, the well-known
# local-to-global map in FE methods.

σₖ = get_cell_dof_ids(Uₕ)

# Finally, we can extract the vector of values.

Uₖ = get_cell_dof_values(uₕ)

# Take a look at the type of array
# it is. In Gridap we put negative labels to fixed DOFs and positive to free DOFs,
# thus we use an array that combines σₖ with the two arrays of free and fixed values
# accessing the right one depending on the index. But everything is lazy, only
# computed when accessing the array. Laziness and quasi-immutability are leitmotifs in
# Gridap.

# We can also extract an array that provides at each cell the finite element
# basis in the physical space, which are again fields.

dv = get_cell_shapefuns(Vₕ)
du = get_cell_shapefuns_trial(Uₕ)

# We note that these bases differ from the fact that the first one is of
# test type and the second one of trial type (in the Galerkin method). This information
# is consumed in different parts of the code.



# ## The geometrical model

# From the triangulation we can also extract the cell map, i.e., the geometrical map that takes points
# in the parametric space $[0,1]^D$ (the SEGMENT, QUAD, or HEX in 1, 2, or 3D, resp.) and maps it to the cell in the physical space
# $\Omega$.

ξₖ = get_cell_map(Tₕ)

# The cell map takes at each cell points in the parametric space and return the
# mapped points in the physical space. Even though this space does not need a global
# definition (nothing has to be solved here), it is continuous across interior faces.

# This cell_map is an `AppliedArray`, one of the essential types in `Gridap`,
# which will be introduced in more detail below. At each cell, it provides the
# `Field` that maps parametric cells to physical cells.

# The node coordinates can be extracted from the triangulation, returning a
# global array of `Point`. You can see that such array is stored using Cartesian
# indices instead of linear indices. It is more natural for Cartesian meshes.

X = get_node_coordinates(Tₕ)

# You can also extract a cell-wise array that provides the node indices per cell

ctn = get_cell_node_ids(Tₕ)

# or the cell-wise nodal coordinates, combining the previous two arrays

_Xₖ = get_cell_coordinates(Tₕ)

# ## A low-level definition of the cell map

# Now, let us create the geometrical map almost from scratch, in order to
# get familiarised with the `Gridap` internals.
# First, we start with the reference topology of the representation that we
# will use for the geometry. In this example, we consider that the geometry
# is represented with a bilinear map, and we use a scalar-valued FE space to
# combine the nodal coordinate values which is a Lagrangian first order space.
# To this end, we first need to create a Polytope using an array of dimension D
# with the parameter HEX_AXIS. This represents an n-cube of dimension D. Then,
# this is used to create the scalar first order Lagrangian reference FE.
# It is not the purpose of this tutorial to describe the `ReferenceFE` in Gridap.

pol = Polytope(Fill(HEX_AXIS,D)...)
reffe_g = LagrangianRefFE(Float64,pol,1)

# Next, we extract the basis of shape functions for this Reference FE, which is
# a set of fields, as many as shape functions. We note that these fields have
# as domain the parametric space $[0,1]^D$. Thus, they can readily be evaluated for
# points in the parametric space.

ϕrg = get_shapefuns(reffe_g)

# Now, we create a global cell array that has the same reference FE basis for all
# cells. We can do this efficiently with `FillArrays` package and its `Fill` method, it
# only stores the value once and returns it for whatever index.

ϕrgₖ = Fill(ϕrg,num_cells(model))

# We can use the following `LocalToGlobalArray` in `Gridap` that returns a
# lazy array of arrays of `Point`, i.e., the coordinates of the nodes per
# each cell.

# OLD Xₖ = LocalToGlobalArray(ctn,X)

Xₖ = lazy_map(Broadcasting(Reindex(X)),ctn)

#

@test Xₖ == _Xₖ == get_cell_coordinates(Tₕ) # check

# Even though inline evaluations in your code editor
# (or if you just call the @show method) are showing the full matrix, don't get
# confused. This is because this method is evaluating the array at all indices and
# collecting and printing the result. In practical runs, this array, as many other in
# `Gridap`, is lazy. We only compute its entries for a given index on demand, by
# accessing to the pointer array `lcn` and extract the values in `X`.

# Next, we can compute the geometrical map as the combination of these shape
# functions in the parametric space with the node coordinates (at each cell)

#
# OLD lc = Gridap.Fields.LinComValued()
# OLD lcₖ = Fill(lc,num_cells(model))
# OLD ψₖ = apply(lcₖ,ϕrgₖ,Xₖ)

# Note that since we use the same kernel for all cells, we don't need to build the array of kernels
# `lcₖ`, we can simply write

ψₖ = lazy_map(linear_combination,Xₖ,ϕrgₖ)

#

@test lazy_map(evaluate,ψₖ,qₖ) == lazy_map(evaluate,ξₖ,qₖ) # check

# We have re-computed (in a low-level way) the geometrical map. First, we have
# created a (constant) array with the kernel `LinComValued`. We have internally
# defined many different `Kernel`s in `Gridap`, which act on `Field`s. The one here
# takes the reference FE basis functions and linearly combines them using the
# nodal coordinates in the physical space (which are `VectorValued`). This is the
# mathematical definition of the geometrical map in FEs!

# At this point, let us define what `apply` is doing. It creates an `AppliedArray`,
# which is one of the essential components of `Gridap`. An `AppliedArray` is a
# lazy array that applies arrays of kernels (operations) over array(s). These operations are
# only computed when accessing the corresponding index, thus lazy. This way,
# we are implementing expression trees of arrays. On top of this, these arrays
# can be arrays of `Field`, as `ϕrgₖ` above. These lazy arrays are implemented
# in an efficient way, creating a cache for the result of evaluating it a a given
# index. This way, the code is performant, and does not involve allocations when
# traversing these arrays. It is probably a good time to take a look at `AppliedArray`
# and the abstract API of `Kernel` in `Gridap`.

# It is good to mention that `apply(k,a,b)`` is equivalent to
# map((ai,bi)->apply_kernel(k,ai,bi),a,b) but with a lazy result instead of a
# plain Julia array.

# With this, we can compute the Jacobian (cell-wise).
# The Jacobian of the transformation is simply its gradient.
# The gradient in the parametric space can be computed as a gradient of the
# global array defined before, or taking the gradient and filling the array

∇ϕrg  = Broadcasting(∇)(ϕrg) # = evaluate(Broadcasting(∇),ϕrg) = broadcast(∇,ϕrg)
∇ϕrgₖ = Fill(∇ϕrg,num_cells(model))

#
lazy_map(evaluate,∇ϕrgₖ,qₖ)
# PENDING
#@test lazy_map(evaluate,∇ϕrgₖ,qₖ) == evaluate(∇(ϕrgₖ),qₖ)

#

#J = apply(lc,∇ϕrgₖ,Xₖ)
J = lazy_map(linear_combination,Xₖ,∇ϕrgₖ)
# Why evaluate(Broadcasting(∇)(ξₖ),qₖ) fails?
#
#
evaluate(lazy_map,evaluate,J,qₖ)
# PENDING Why evaluate(lazy_map(∇,ξₖ),qₖ) not working?
# @test all(evaluate(J,qₖ) .≈ evaluate(∇(ξₖ),qₖ))

# ## A low-level definition of FE space bases

# We proceed as before, creating the reference FE, the reference basis, and the
# corresponding constant array.

pol = Polytope(Fill(HEX_AXIS,D)...)
reffe = LagrangianRefFE(T,pol,order)

ϕr = get_shapefuns(reffe)
ϕrₖ = Fill(ϕr,num_cells(model))

# As stated in FE theory, we can now define the shape functions in the physical
# space, which are conceptually $ϕ(x) = ϕr_K(X)∘ξ_K^{-1}(x)$. We provide a kernel
# for this, `AddMap`. First, we create this kernel,
# and then we apply it to the basis in the parametric space and
# the geometrical map

# TRIGGERS ERRORS (PENDING)
ϕₖ = lazy_map(Broadcasting(∘),ϕrₖ,lazy_map(inverse_map,ξₖ))

# PENDING
# P1. map = Gridap.Fields.AddMap()
# P2. ϕₖ = apply(map,ϕrₖ,ξₖ)
# P3. @test ϕₖ === attachmap(ϕrₖ,ξₖ)

# Again, the result is an `AppliedArray` that provides a `Field` at each cell.

# We note that when using ref FEs in the parametric space, the points in which
# we will evaluate the function are also in the parametric space (quadrature in
# $[0,1]^D$). Thus, the geometrical map is not really needed for this evaluation.
# However, it is essential
# when computing its gradient, later on. There is another path in Gridap, which
# defines reference FEs in the physical space, but it won't be considered here.

# Even though this is not essential for this tutorial, we note that we can
# create a cell basis, a `GenericCellBasis` struct, which represents our
# shape functions. It takes ϕₖ (the shape functions), the cell map ξₖ and some metadata,
# namely
# the trial style (the first argument, true means it is a trial FE space, test FE space otherwise
# in the Galerkin method parlance),
# the reference trait (the last Val{true}, true means FEs define in the reference
# FE space, the most common case, false means FEs with DOFs defined in the physical
# space).

bₖ = Gridap.FESpaces.FEBasis(ϕrₖ,Tₕ,Gridap.FESpaces.TrialBasis(),ReferenceDomain())

# We can check that the basis we have created return the same values as the
# one obtained with high-level APIs

@test lazy_map(evaluate,get_cell_data(dv),qₖ) == lazy_map(evaluate,get_cell_data(bₖ),qₖ)

# There are some objects in `Gridap` that are nothing but a lazy array plus
# some metadata. Another example could be an array of arrays of points like `q`.
# `q` points are in the reference space. You could consider creating `CellPoints`
# with a trait that tells you whether these points are in the parametric or
# physical space, and use dispatching based on that. E.g., if you have a reference
# FE in the reference space, you can easily evaluate in the parametric space,
# but you should map the points to the physical space first with `ξₖ` when
# dealing with FEs in the physical space.

# Another salient feature of Gridap is that for these finite element bases,
# we can readily compute the gradient as ∇(ϕₖ), which internally is implemented
# as a lazy array as follows

# The computation of the gradient of FE shape functions in the physical space
# would require to create a Kernel with the inv() to create the inv(J)
# that is needed for the computation of derivatives in the physical space
# but we have merged all the operations in the PhysGrad() kernel.

# grad = Gridap.Fields.Valued(Gridap.Fields.PhysGrad())
# ∇ϕrₖ = Fill(Gridap.Fields.FieldGrad(ϕr),num_cells(Tₕ))
# ∇ϕₖ = apply(grad,∇ϕrₖ,J)

∇ϕr  = Broadcasting(∇)(ϕr)
∇ϕrₖ = Fill(∇ϕr,num_cells(Tₕ))
∇ϕₖ  = lazy_map(Broadcasting(push_∇),∇ϕrₖ,ξₖ)

∇ϕrᵀ  = Broadcasting(∇)(transpose(ϕr))
∇ϕrₖᵀ = Fill(∇ϕrᵀ,num_cells(Tₕ))
∇ϕₖᵀ  = lazy_map(Broadcasting(push_∇),∇ϕrₖᵀ,ξₖ)
#
lazy_map(evaluate,∇ϕₖ,qₖ)
# PENDING
# @test evaluate(∇ϕₖ,qₖ) == evaluate(∇(ϕₖ),qₖ)
@test lazy_map(evaluate,∇ϕₖ,qₖ) == lazy_map(evaluate,get_cell_data(∇(dv)),qₖ)

# We can now evaluate both the CellBasis and the array of physical shape functions,
# and check we get the same.

#@test evaluate(∇ϕₖ,qₖ) == evaluate(∇(bₖ),qₖ) == evaluate(∇(dv),qₖ)

#@test evaluate(∇ϕₖ,qₖ) == evaluate(∇(dv),qₖ)


# ## A low-level definition of the FE function

# Let us explore this FE function.
# Now, let us create uₖ from scratch, in order to understand how
# Gridap works internally, and why we say that Gridap design strongly relies on
# lazy (evaluation of) arrays.

# In fact, we can create uₖ by our own with the ingredients we already have. uₖ is an
# array that linearly combines the basis ϕₖ and the dof values Uₖ at each cell.
# The kernel that does this in Gridap is `LinComValued()`. So, we create a
# this kernel and create
# and applied array with all these ingredients. As above, it is a lazy array
# that will return the shape functions at each cell in the physical space

uₖ_own = lazy_map(linear_combination,Uₖ,ϕrₖ)

# PENDING: I cannot use ϕₖ in the line right above
# lc = Gridap.Fields.LinComValued()
# uₖ = apply(lc,ϕₖ,Uₖ)

# We can check that we get the same results as with uₖ

@test lazy_map(evaluate,uₖ_own,qₖ) == lazy_map(evaluate,uₖ,qₖ)

# Now, since we can apply the gradient over this array

# PENDING
# gradient(uₖ)

# or compute it using low-level methods, as a linear combination
# of ∇(ϕₖ) instead of ϕₖ
∇uₖ_own = lazy_map(linear_combination,Uₖ,∇ϕₖ)

# PENDING
# aux = ∇(uₖ)

∇uₖ = get_cell_data(∇(uₕ))


# We can check we get the expected result

@test all(lazy_map(evaluate,∇uₖ,qₖ) .≈ lazy_map(evaluate,∇uₖ_own,qₖ))

# ## A low-level implementation of the residual integration and assembly

# We have the array uₖ that returns the finite element function uₕ at
# each cell, and its gradient ∇uₖ.
# Let us consider now the integration of (bi)linear forms. The idea is to
# compute first the following residual for our random function uₕ

intg = get_cell_data(∇(uₕ)⋅∇(dv))

# but we are going to do it using low-level methods.

# First, we create an array that for each cell returns the dot operator

# dotop = Gridap.Fields.FieldBinOp(dot)
# dotopv = Gridap.Fields.Valued(dotop)
# Iₖ = apply(⋅,∇uₖ,∇ϕₖ)

Iₖ = lazy_map(Broadcasting(Operation(⋅)),∇uₖ,∇ϕₖ)

# Next we consider a lazy `AppliedArray` that applies the `dot_ₖ` array of
# operations (binary operator) over the gradient of the FE function and
# the gradient of the FE basis in the physical space

@test lazy_map(evaluate,intg,qₖ) == lazy_map(evaluate,Iₖ,qₖ)

# Now, we can finally compute the cell-wise residual array, which using
# the high-level `integrate` function is

res = integrate(∇(uₕ)⋅∇(dv),Qₕ)

# In a low-level, what we do is to apply (create a `AppliedArray`)
# the `IntKernel` over the integrand evaluated at the integration
# points, the weights, and the Jacobian evaluated at the integration points

Jq = lazy_map(evaluate,J,qₖ)
intq = lazy_map(evaluate,Iₖ,qₖ)
iwq = lazy_map(IntegrationMap(),intq,Qₕ.cell_weight,Jq)

@test all(res .≈ iwq)

# The result is the cell-wise residual (previous to assembly). This is a lazy
# array but you could collect the element residuals if you want

collect(iwq)

# Alternatively, we could use the high-level API that creates a `LinearFETerm`
# that is the composition of a lambda-function or
# [anonymous function](https://docs.julialang.org/en/v1/manual/functions/#man-anonymous-functions-1)
# with the bilinear form, triangulation and quadrature

#blf(u,v) = ∇(u)⋅∇(v)
#term = LinearFETerm(blf,Tₕ,Qₕ)
# cellvals = get_cell_residual(term,uₕ,dv)

cellvals = ∫( ∇(dv)⋅∇(uₕ) )*Qₕ


# and check that we get the same residual as the one defined above
@test cellvals == iwq

# ## Assembling a residual

# Now, we need to assemble these cell-wise (lazy) residual contributions in a
# global (non-lazy) array. With all this, we can assemble our vector using the
# cell_values and the assembler.
# Let us create a standard assembly struct for the finite element spaces at
# hand. It will create a vector of size global number of dofs, and a
# `SparseMatrixCSC` in which we can add contributions.

assem = SparseMatrixAssembler(Uₕ,Vₕ)

# We create a tuple with 1-entry arrays with the cell-wise vectors and cell ids.
# If we had additional terms, we would have more entries in the array.
# You can take a look at the `SparseMatrixAssembler` struct.

cellids = get_cell_to_bgcell(Tₕ) # == identity_vector(num_cells(trian))

rs = ([iwq],[cellids])
b = allocate_vector(assem,rs)
assemble_vector!(b,assem,rs)

# ## A low-level implementation of the Jacobian integration and assembly

# After computing the residual, we use similar ideas for the Jacobian.
# The process is the same as above, so it does not require more explanations

int = lazy_map(Broadcasting(Operation(⋅)),∇ϕₖ,∇ϕₖᵀ)
@test all(collect(lazy_map(evaluate,int,qₖ)) .== collect(lazy_map(evaluate,get_cell_data(∇(du)⋅∇(dv)),qₖ)))

intq = lazy_map(evaluate,int,qₖ)
Jq = lazy_map(evaluate,J,qₖ)
iwq = lazy_map(IntegrationMap(),intq,Qₕ.cell_weight,Jq)

jac = integrate(∇(du)⋅∇(dv),Qₕ)
@test collect(iwq) == collect(jac)

rs = ([iwq],[cellids],[cellids])
A = allocate_matrix(assem,rs)
A = assemble_matrix!(A,assem,rs)

# Now we can obtain the free dofs and add the solution to the initial guess

x = A \ b
uf = sol = get_free_values(uₕ) - x
ufₕ = FEFunction(Uₕ,uf)

#

@test sum(integrate((u-ufₕ)*(u-ufₕ),Qₕ)) <= 10^-8
