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
# the code snippets are quite subtle/technical and may require a deeper exploration
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

## Exploring our first `CellField` objects, namely `FEBasis` objects, and its evaluation

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
# points in physical space. At this point, the reader may want to observe which object results from
# the evaluation of, e.g., `dv_at_Qₕ`, at a different set points for each cell (e.g. by building
# its own array of arrays of `Points`).

# Going back to our example, any entry of `dv_at_Qₕ` is a rank-2 array of size 4x4 that provides in # position `[i,j]` the i-th test shape function at the j-th quadrature rule evaluation point.
# On the other hand, any entry of `du_at_Qₕ` is a rank-3 array of size `4x1x4` that provides in
# position `[i,1,j]` the i-th trial shape function at the j-th quadrature point. The reader might
# be wondering why the rank of these two arrays are different. The rationale is that, by means of
# a broadcasted `*` operation of these two arrays, we can get a 4x4x4 array where the `[i,j,k]`
# entry stores the product of the i-th test and j-th trial functions, both evaluated at the k-th
# quadrature point. If we sum over the $k$-index, we obtain the necessary data to compute the cell-local
# matrix that
# we assemble into the global matrix in order to get a mass matrix (neglecting the strong
# imposition of Dirichlet boundary conditions). For those readers more used to traditional
# finite element codes, the broadcast followed by the sum over k, provides the data required
# in order to implement the following triple standard for-nested loop:

#   M[:,:]=0.0
#   Loop over quadrature points k
#     detJ_wk=det(J)*w[k]
#     Loop over shape test functions i
#       Loop over shape trial functions j
#            M[i,j]+=shape_test[i,k]*shape_trial[i,k]*detJ_wk

# where det(K) represents the determinant of the reference-physical mapping of the current cell,
# and w[k] the quadrature rule weight corresponding to the k-th evaluation point.
# Using Julia built-in support for broadcasting, we can vectorize the full operation, and get much
# higher performance.

# The highest-level possible way of performing the aforementioned broadcasted `*` is by building
# a "new" `CellField` instance by multiplying the two `FEBasis` objects, and then evaluating the
# resulting object at the points in `Qₕ_cell_point`. This is something common in `Gridap`. One can
# create new `CellField` objects out of existing ones, e.g., by performing operations among them, or
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

# Recall from above that `CellField` objects are also `CellDatum` objects. Thus, one can
# use the `get_cell_data` generic function to extract, in an array, the collection of
# quantities, one per each cell of the triangulation, out of them. As one may expect,
# in the case of our `FEBasis` objects `dv` and `du` at hand, `get_cell_data` returns
# a (cell) array of arrays of `Field` objects, i.e., the cell-local shape basis functions:

dv_array = get_cell_data(dv)
du_array = get_cell_data(du)

@test isa(dv_array,AbstractVector{<:AbstractVector{<:Field}})
@test isa(du_array,AbstractVector{<:AbstractArray{<:Field,2}})
@test length(dv_array) == num_cells(Tₕ)
@test length(du_array) == num_cells(Tₕ)

# As expected, both `dv_array` and `du_array` are (*conceptually*) vectors (i.e, rank-1 arrays)
# with as many entries as cells. The concrete type of each vector differs, though, i.e.,
# `Fill` and `LazyArray`, resp. (We will come back to `LazyArray`s below,
# as they play a fundamental role in the way in which the finite element method is implemented
# in `Gridap`.) For each cell, we have arrays of `Field` objects. Recall from above that `Map` and # `Field` (with `Field` a subtype of `Map`), and `CellDatum` and `CellField` (with `CellField` a
# subtype of `CellDatum`) and the associated type hierarchies, are fundamental in `Gridap` for the # implementation of variational methods in finite-dimensional spaces. `Field` conceptually
# represents a physical (scalar, vector, or tensor) field. `Field` objects can be evaluated at
# single `Point` objects (or at an array of them in one shot), and they return scalars
# (i.e., a sub-type of Julia `Number`), `VectorValue`, or `TensorValue` objects (or an array of
# them, resp.)

# In order to evaluate a `Field` object at a `Point` object, or at an array of `Points`, we can use
# the `evaluate` generic function. For example, the following statement

ϕ₃ = dv_array[1][3]
evaluate(ϕ₃,[Point(0,0),Point(1,0),Point(0,1),Point(1,1)])

# evaluates the 3rd test shape function of the local space of the first cell at the 4
# vertices of the cell (recall from above that, for the implementation of Lagrangian finite
# elements being used in this tutorial, shape functions are though to be evaluated at point
# coordinates expressed in the parametric space of the reference cell).
# As expected, ϕ₃ evaluates to one at the 3rd vertex of the cell, and
# to zero at the rest of vertices, as ϕ₃ is the shape function associated to the
# Lagrangian node/dof located at the 3rd vertex. We can also evaluate all shape functions of
# the local space of the first cell (i.e., an array of `Field`s) at once at an array of `Points`

ϕ = dv_array[1]
evaluate(ϕ,[Point(0,0),Point(1,0),Point(0,1),Point(1,1)])

# As expected, we get the Identity matrix, as the shape functions of the local space have,
# by definition, the Kronecker delta property.

# However, and here comes one of the main take-aways of this tutorial, in `Gridap`,
# (cell-wise) arrays of `Fields` (or arrays of `Fields`) are definitely NOT conceived to
# be evaluated following the approach that we used in the previous examples, i.e.,
# by manually extracting the `Field` (array of `Field`s) corresponding to a cell,
# and then evaluating it (them) at a given set of `Point`s. Instead, one uses the `lazy_map`
# generic function, which combined with the `evaluate` function, represents the operation
# of walking over all cells, and evaluating the fields, cell by cell, as a whole. This is
# illustrated in the following piece of code:

dv_array_at_qₖ = lazy_map(evaluate,dv_array,qₖ)
du_array_at_qₖ = lazy_map(evaluate,du_array,qₖ)

# We note that the results of these two expressions are equivalent to the ones of `evaluate(dv,
# Qₕ_cell_point)` and `evaluate(du,Qₕ_cell_point)`, resp. (check it!) In fact, these latter
# two expressions translate under the hood into the calls to `lazy_map` above. These calls to
# `lazy_map` return an array of the same length of the input arrays, with their i-th entry
# conceptually defined, e.g., as `evaluate(du_array[i],qₖ[i])` in the case of the second array.
# To be "conceptually defined as" does not mean that they are actually computed as
# `evaluate(du_array[i],qₖ[i])`. Indeed they don't, it would not be high performant.

# You might be now wondering which is the main point behind `lazy_map`.
# `lazy_map` turns to be cornerstone in `Gridap`. (At this point, you may execute `methods
# (lazy_map)` to observe that a large amount of programming logic is devoted to it.)
# Let us try to answer it more abstractly now. However, this will be revisited along the tutorial
# with additional examples.

# 1. To keep memory demands at low levels,
#    `lazy_map` NEVER returns an array that stores the result at all cells at once.
#    In the two examples above, this is achieved using `Fill` arrays. However, this is
#    only possible in very particular scenarios (see discussion above) . In more general cases,
#    the array resulting from `lazy_map` does not have the same entry in all cells. In such cases,
#    `lazy_map` returns a `LazyArray`, which is another
#    essential component of `Gridap`. In a nutshell, a  `LazyArray` is an
#    array that applies, entry-wise, arrays of operations over array(s). These operations are
#    only computed when accessing the corresponding index, thus the name lazy.
#    Besides, the entries of these are computed in an efficient way, using a set of mechanisms
#    that will be illustrated below with examples.
#
# 2. Apart from `Function` objects, such as `evaluate`, `lazy_map` can also be
#    combined with other objects, e.g., `Map`s. For example, `Broadcasting(*)` presented above.
#    As there are `Map`s that can be applied to `Field`s (or arrays of `Fields`) to build new
#    `Field`s (or arrays of `Fields`), the recursive application of `lazy_map` let us build
#    complex operation trees among arrays of `Field`s as the ones required for the implementation of
#    variational forms. While building these trees, by virtue of Julia support
#    for multiple type dispatching, there are plenty of opportunities for
#    optimization by changing the order in which the operations are performed. These optimizations
#    typically come in the form of a significant saving of FLOPs, by exploiting the particular
#    properties of the `Field`s at hand, or into higher
#    granularity for vectorized array operations
#    when the expressions are actually evaluated.
#    Indeed, the arrays that one usually obtains from `lazy_map` are not the trivial
#    `LazyArray`s that one would expect from a naive combination of the arguments to `lazy_map`.
#

# ## Exploring another type of `CellField` objects, FE functions

# Let us now work with another type of `CellField` objects, the ones that are used
# to represent an arbitrary element of a global FE space of functions, i.e., a FE function.
# A global FE function can be understood conceptually as a collection of `Field`s, one per each
# cell of the triangulation. The `Field` corresponding to a cell represents the restriction
# of the global FE function to the cell (Recall that in finite elements, global functions
# are defined piece-wise on each cell.) As we did in the previous section, we will explore,
# at different levels, how FE functions are evaluated. However, we will dig deeper into this
# by illustrating some the aforementioned mechanisms on which `LazyArray` relies in order
# to efficiently implement the entry-wise application of an operation (or array of operations)
# to a set of input arrays.

# Let us now build a FE function belonging to the global trial space of functions Uₕ,
# with [rand](https://docs.julialang.org/en/v1/stdlib/Random/#Base.rand) free dof values.
# Using `Gridap`'s higher-level API, this can be achieved as follows

uₕ = FEFunction(Uₕ,rand(num_free_dofs(Uₕ)))

# As expected from the discussion above, the returned object is a `CellField` object.

@test isa(uₕ,CellField)

# Thus, we can, e.g., query the value of its `DomainStyle` trait, that turns to be
# `ReferenceDomain`

@test DomainStyle(uₕ) == ReferenceDomain()

# Thus, in order to evaluate the `Field` object that represents the restriction of the
# FE function to a given cell, we have to provide `Point`s in the parametric space
# of the reference cell, and we get the value of the FE function at the corresponding
# mapped `Point` in the physical domain. This should not come as a surprise as we have
# that: (1) the restriction of the FE function to a given cell is mathematically defined
# as a linear combination of the local shape functions of the cell (with coefficients given
# by the values of the dofs at the cell). (2) As observed in the previous section, the shape
# functions are such that they have to be evaluated at `Point`s in the parametric space of
# of the reference cell. This property is thus transferred to the FE function.

# As FE functions are `CellField` objects, we can evaluate them at `CellPoint` objects.
# Let us do it at the points within `Qₕ_cell_point` (see above for a justification of why
# this is possible):

uₕ_at_Qₕ = evaluate(uₕ,Qₕ_cell_point)

# For the first time in this tutorial, we have obtained a cell array of type `LazyArray`
# from evaluating a `CellField` at a `CellPoint`.

@test isa(uₕ_at_Qₕ,LazyArray)

# This makes sense as a finite element function restricted to a cell is, in general,
# different in each cell, i.e., it evaluates to different values at the
# quadrature rule evaluation points. In other words, the `Fill` array optimization that
# was performed for the evaluation of the cell-wise local shape functions `dv` and `du`
# does not apply here, and a `LazyArray` has to be used instead.

# Alhough it is hard to understand the full concrete type name of `uₕ_at_Qₕ` at this time

print(typeof(uₕ_at_Qₕ))

# we will dissect `LazyArray`s up to an extent that hopefully after we complete this section we will
# have a better grasp of it. By now, the most important thing for you to keep in mind is that
# `LazyArray`s objects encode a recipe to produce its entries just-in-time when they are accessed.
# They NEVER store all of its entries at once. Even if the expression `uₕ_at_Qₕ` typed in the REPL,
# (or inline evaluations in your code editor) show the array with all of its entries at once, don't
# get confused. This is because the Julia REPL is evaluating the array at all indices and
# collecting the result just for printing purposes.

# Just as we did with `FEBasis`, we can extract an array of `Field` objects out of `uₕ_at_Qₕ`, as
# `uₕ_at_Qₕ` is also a `CellBasis` object.

uₕ_array = get_cell_data(uₕ)

# As expected, `uₕ_array` is (conceptually) a vector (i.e., rank-1 array) of `Field` objects.

@test isa(uₕ_array,AbstractVector{<:Field})

# Its concrete type is, though, `LazyArray`

@test isa(uₕ_array,LazyArray)

# with full name, as above, of a certain complexity (to say the least):

print(typeof(uₕ_array))

# As mentioned above, `lazy_map` returns `LazyArray`s in the most general scenarios.
# Thus, it is reasonable to think that `get_cell_data(uₕ)` returns an array that has
# been built via `lazy_map`. (We advance now that this is indeed the case.)
# On the other hand, as `uₕ_array` is (conceptually)
# a vector (i.e., rank-1 array) of `Field` objects, this also tells us that the
# `lazy_map`/`LazyArray` pair does not only play a fundamental role in the evaluation
# of (e.g., cell)
# arrays of `Field`s on (e.g., cell) arrays of arrays of `Point`s, but also in building new cell
# arrays of `Field`s (i.e., the local restriction of a FE function to each cell) out of
# existing ones
# (i.e., the cell array with the local shape functions). In the words of the previous section,
# we can use `lazy_map` to build complex operation trees among arrays of `Field`s, as required
# by the computer implementation of variational methods.

# The key question now is: what is the point behind `get_cell_data(uₕ)` returning a `LazyArray` of
# `Field`s, and not just a plain Julia array of `Field`s? At the end of the day, `Field` objects
# themselves have very low memory demands, they only need to hold the necessary
# information to encode their action (evaluation) on a `Point`/array of `Point`s.
# This is in contrast to the evaluation of (e.g., cell) arrays of `Field`s
# (or arrays of `Field`s) at an array of `Point`s, which does consume a significantly
# larger amount of memory (if all entries are to be stored at once in memory, and not by demand).
# The short answer is higher performance. Using `LazyArray`s to encode operation trees among
# cell arrays of `Field`s, we can apply optimizations when evaluating these operation trees
# that would be deactivated if we just computed a plain array of `Field`s.
# If all this sounds quite abstract,
# (most probably it does), we are going to dig into this further in the rest of the section.

# As mentioned above, `uₕ_array` can be conceptually seen as an array of `Field`s. Thus, if we access to a particular entry of it, we should get a `Field` object. (Although possible, this is not the way in which `uₕ_array` is conceived to be used, as was also mentioned in the previous section.) This is indeed confirmed when accessing to, e.g., the third entry of `uₕ_array`:

uₕ³ = uₕ_array[3]
@test isa(uₕ³,Field)

# The concrete type of `uₕ³` is `LinearCombinationField`. This type represents a `Field`
# defined as a linear combination of an existing vector of `Field`s. This sort of `Field`s
# can be built using the `linear_combination` generic function.
# Among its methods, there is one which takes (1) a vector of scalars (i.e., Julia `Number`s)
# with the coefficients of the expansion and (2) a vector of `Field`s as its two arguments,
# and returns a `LinearCombinationField` object. As mentioned above, this is the exact
# mathematical definition of a FE function restricted to a cell.

# Let us manually build uₕ³. In order to do so, we can first use the `get_cell_dof_values` function, which extracts out of uₕ a cell array of arrays with the dof values of uₕ restricted to all cells of the triangulation (defined from a conceptual point of view).

Uₖ = get_cell_dof_values(uₕ)

# (The returned array turns to be of concrete type `LazyArray`, again to keep memory demands low, but let us skip this detail for the moment.) If we restrict `Uₖ` and `dv_array` to the third cell

Uₖ³ = Uₖ[3]
ϕₖ³ = dv_array[3]

# we get the two arguments that we need to invoke `linear_combination` in order to build our manually built version of uₕ³

manual_uₕ³ = linear_combination(Uₖ³,ϕₖ³)

# We can double-check that `uₕ³` and `manual_uₕ³` are equivalent by evaluating them at the quadrature rule evaluation points, and comparing the result:

@test evaluate(uₕ³,qₖ[3]) ≈ evaluate(manual_uₕ³,qₖ[3])

# Following this idea, we can go even further and manually build a plain Julia vector of `LinearCombinationField` objects as follows:

manual_uₕ_array = [linear_combination(Uₖ[i],dv_array[i]) for i=1:num_cells(Tₕ)]

# And we can (lazily) evaluate this manually-built array of `Field`s at a cell array of arrays of `Point`s (i.e., at `qₖ`) using `lazy_map`:

manual_uₕ_array_at_qₖ = lazy_map(evaluate,manual_uₕ_array,qₖ)

# The entries of the resulting array are equivalent to those of the array that we obtained from `Gridap` automatically, i.e., `uₕ_at_Qₕ`

@test any( uₕ_at_Qₕ .≈ manual_uₕ_array_at_qₖ )

# However, and here it comes the key of the discussion, the concrete types of `uₕ_at_Qₕ` and `manual_uₕ_array_at_qₖ` do not match.

@test typeof(uₕ_at_Qₕ) != typeof(manual_uₕ_array_at_qₖ)

# This is because `evaluate(uₕ,Qₕ_cell_point)` does not follow the (naive) approach that we followed to build `manual_uₕ_array_at_qₖ`, but it instead calls `lazy_map` under the hood as follows

uₕ_array_at_qₖ = lazy_map(evaluate,uₕ_array,qₖ)

# Now we can see that the types of `uₕ_array_at_qₖ` and `uₕ_at_Qₕ` match:

@test typeof(uₕ_array_at_qₖ) == typeof(uₕ_at_Qₕ)

# Therefore, why `Gridap` does not build `manual_uₕ_array_at_qₖ`? what's wrong with it? Let us first try to answer this quantitatively. Let us assume that we want to sum all entries of a `LazyArray`. In the case of `LazyArray`s of arrays, this operation is only well-defined if the size of the arrays of all antries matches. This is the case of the `uₕ_array_at_qₖ` and `manual_uₕ_array_at_qₖ` arrays, as we have the same quadrature rule at all cells. We can write this function following the `Gridap` internals' way.

function smart_sum(a::LazyArray)
  cache=array_cache(a)             # Create cache out of a
  sum=copy(getindex!(cache,a,1))   # We have to copy the output
                                   # from get_index! to avoid array aliasing
  for i in 2:length(a)
    ai = getindex!(cache,a,i)      # Compute the i-th entry of a
                                   # re-using work arrays in cache
    sum .= sum .+ ai
  end
  sum
end

# The function uses the so-called "cache" of a `LazyArray`. In a nutshell, this cache can be thought as a place-holder of work arrays that can be re-used among different evaluations of the entries of the `LazyArray` (e.g., the work array in which the result of the computation of an entry of the array is stored.) This way, the code is more performant, as the cache avoids that these work arrays are created repeteadly when traversing the `LazyArray` and computing its entries. It turns out that `LazyArray`s are not the only objects in `Gridap` that (can) work with caches. `Map` and `Field` objects also provide caches for reusing temporary storage among their repeated evaluation on different arguments of the same types. (For the eager reader, the cache can be obtained out of a `Map`/`Field` with the `return_cache` abstract method; see also `return_type`, `return_value`, and `evaluate!` functions of the abstract API of `Map`s). When a `LazyArray` is created out of objects that in turn rely on caches (e.g., a `LazyArray` with entries defined as the entry-wise application of a `Map` to two `LazyArrays`), the caches of the latter objects are also handled by the former object, so that this scheme naturally accommodates top-down recursion, as per-required in the evaluation of complex operation trees among arrays of `Field`s, and their evaluation at a set of `Point`s. We warn the reader this is a quite complex mechanism. The reader is encouraged to follow with a debugger, step by step, the execution of the `smart_sum` function with the `LazyArray`s built above in order to gain some familiarity with this mechanism.

# If we @time the `smart_sum` function with `uₕ_array_at_qₖ` and `manual_uₕ_array_at_qₖ`

smart_sum(uₕ_array_at_qₖ)        # Execute once before to neglect JIT-compilation time
smart_sum(manual_uₕ_array_at_qₖ) # Execute once before to neglect JIT-compilation time
@time begin
        for i in 1:100_000
         smart_sum(uₕ_array_at_qₖ)
        end
      end
@time begin
        for i in 1:100_000
          smart_sum(manual_uₕ_array_at_qₖ)
        end
      end

# we can observe that the array returned by `Gridap` can be summed in significantly less time, using significanly less allocations. [WHY THE SECOND PIECE OF CODE REQUIRES A NUMBER OF ALLOCATIONS THAT GROWS WITH THE NUMBER OF CELLS? I CAN UNDERSTAND THAT THE CACHE ARRAY of `manual_uₕ_array_at_qₖ` REQUIRES MORE MEMORY IN ABSOLUTE TERMS, BUT I AM NOT ABLE TO SEE WHY IT GROWS WITH THE NUMBER OF CELLS!!!!]

# Let us try to answer the question now qualitatively. In order to do so, we can take a look at the full names of the types of both `LazyArray`s. `LazyArray`s have four type parameters, referred to as `G`, `T`, `N`, and `F`. `G` is the type of the array with the entry-wise operations (e.g., a `Function` object, a `Map` or a `Field`) to be applied, `T` is the type of the elements of the array, and `N` its rank. Finally, `F` is a `Tuple` with the types of the arrays to which the operations are applied in order to obtain the entries of the `LazyArray`. We can use the following function to pretty-print the `LazyArray` data type name in a more human-friendly way, while taking into account that types in `F` may be in turn `LazyArray`s recursively:

function print_lazy_array_type_parameters(indentation,a::Type{LazyArray{G,T,N,F}}) where {G,T,N,F}
   println(indentation,"G: $(G)")
   println(indentation,"T: $(T)")
   println(indentation,"N: $(N)")
   for (i,f) in enumerate(F.parameters)
     if (isa(f,Type{<:LazyArray}))
       println(indentation, "F[$i]: LazyArray")
       print_lazy_array_type_parameters(indentation*"   ",f)
     else
       println(indentation,"F[$i]: $(f)")
     end
   end
end

print_lazy_array_type_parameters("",typeof(uₕ_array_at_qₖ))
print_lazy_array_type_parameters("",typeof(manual_uₕ_array_at_qₖ))

# We can observe from the output of these calls the following:
# 1. `uₕ_array_at_qₖ` is a `LazyArray` whose entries are defined as the result of applying a `Fill` array of `LinearCombinationMap{Colon}` `Map`s (G) to a `LazyArray` (F[1]) and a `Fill` array (F[2]). The first array provides the FE function dof values restricted to each cell, and the second the local basis shape functions evaluated at the quadrature points. As the shape functions in physical space have the same values in all cells at the corresponding mapped points in physical space, there is no need to re-evaluate them at each cell, we can evaluate them only once. And this is what the second `Fill` array stores as its unique entry, i.e., a matrix M[i,j] defined as the value of the j-th `Field` (i.e., shape function) evaluated at the i-th `Point`. *This is indeed the main optimization that `lazy_map` applies compared to our manual construction of `uₕ_array_at_qₖ`.* It worths noting that, if `v` denotes the linear combination coefficients, and `M` the matrix resulting from the evaluation of an array of `Fields` at a set of `Points`, with M[i,j] being the value of the j-th `Field` evaluated at the i-th point, the evaluation of `LinearCombinationMap{Colon}` at `v` and `M` returns a vector `w` with w[i] defined as w[i]=sum_k v[k]*M[i,k], i.e., the FE function evaluated at the i-th point. `uₕ_array_at_qₖ` handles the cache of `LinearCombinationMap{Colon}` (which holds internal storage for `w`) and that of the first `LazyArray` (F[1]), so that when it retrieves the dof values `v` of a given cell, and then applies `LinearCombinationMap{Colon}` to `v` and `M`, it does not have to allocate any temporary working arrays, but re-uses the ones stored in the different caches.
# 2. `manual_uₕ_array_at_qₖ` is also a `LazyArray`, but structured rather differently to `uₕ_array_at_qₖ`. In particular, its entries are defined as the result of applying a plain array of `LinearCombinationField`s (G) to a `Fill` array of `Point`s (F[1]) that holds the coordinates of the quadrature rule evaluation points in the parametric space of the reference cell (wich are equivalent for all cells, thus the `Fill` array). The evaluation of a `LinearCombinationField` on a set of `Point`s ultimately depends on `LinearCombinationMap`. As seen in the previous point, the evaluation of this `Map` requires a vector `v` and a matrix `M`. `v` was built in-situ when building each `LinearCombinationField`, and stored within these instances. However, in contrast to `uₕ_array_at_qₖ`, `M` is not part of `manual_uₕ_array_at_qₖ`, and thus it has to be (re-)computed each time that we evaluate a new `LinearCombinationField` instance on a set of points. This is the main source of difference on the computation times observed. By eagerly constructing our array of `LinearCombinationField`s instead of deferring it until (lazy) evaluation via `lazy_map`, we lost optimization opportunities. We stress that `manual_uₕ_array_at_qₖ` also handles the cache of `LinearCombinationField` (that in turn handles the one of `LinearCombinationMap`), so that we do not need to allocate `M` at each cell, we re-use the space within the cache of `LinearCombinationField`.

# We can also extract the global indices of the DOFs in each cell, the well-known
# local-to-global map in FE methods.

σₖ = get_cell_dof_ids(Uₕ)

# Finally, we can extract the vector of values.



# Take a look at the type of array
# it is. In Gridap we put negative labels to fixed DOFs and positive to free DOFs,
# thus we use an array that combines σₖ with the two arrays of free and fixed values
# accessing the right one depending on the index. But everything is lazy, only
# computed when accessing the array. Laziness and quasi-immutability are leitmotifs in
# Gridap.




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
