# ## Introduction and caveat


# This tutorial is advanced and you only need to go through this if you want to know the internals of `Gridap` and what it does under the hood. Even though you will likely want to use the high-level APIs in `Gridap`, this tutorial will (hopefully) help if you want to become a `Gridap` developer, not just a user. We also consider that this tutorial shows how powerful and expressive the `Gridap` kernel is, and how mastering it you can implement new algorithms not currently provided by the library.

# As any other Gridap tutorial, this tutorial is primarily designed to be executed in a Jupyter notebook environment. However, the usage of a Julia debugger (typically outside of a Jupyter notebook environment), such as, e.g., the Julia REPL-based [`Debugger.jl`](https://github.com/JuliaDebug/Debugger.jl) package, or the one which comes along with the Visual Studio Code (VSCode) extension for the Julia programming language, may help the reader eager to understand the full detail of the explanations given. Some of the observations that come along with the code snippets are quite subtle/technical and may require a deeper exploration of the underlying code using a debugger.

# ## Including Gridap's low-level API

# Let us start including `Gridap` and some of its submodules, to have access to a rich set of not so high-level methods. Note that the module `Gridap` provides the high-level API, whereas the submodules such as, e.g., `Gridap.FESpaces`, provide access to the different parts of the low-level API.

using Gridap
using Gridap.FESpaces
using Gridap.ReferenceFEs
using Gridap.Arrays
using Gridap.Geometry
using Gridap.Fields
using Gridap.CellData
using FillArrays
using Test
using InteractiveUtils

# ## Discrete model and FE spaces set up using high-level API

# We first create the geometry model and FE spaces using the high-level API. In this tutorial, we are not going to describe the geometrical machinery in detail, only what is relevant for the discussion. To simplify the analysis of the outputs, you can consider a 2D mesh, i.e., `D=2` (everything below works for any spatial dimension without any extra complication). In order to make things slightly more interesting, i.e., having non-constant Jacobians, we have considered a mesh that is a stretching of an equal-sized structured mesh.

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

# The next step is to build the global FE space of functions from which we are going to extract the unknown function of the differential problem at hand. This tutorial explores the Galerkin discretization of the scalar Poisson equation. Thus, we need to build H1-conforming global FE spaces. This can be achieved using $C^0$ continuous functions made of piece(cell)-wise polynomials. This is precisely the purpose of the following lines of code.

# First, we build a scalar-valued (`T = Float64`) Lagrangian reference FE of order `order` atop a reference n-cube of dimension `D`. To this end, we first need to create a `Polytope` using an array of dimension `D` with the parameter `HEX_AXIS`, which encodes the reference representation of the cells in the mesh. Then, we create the Lagrangian reference FE using the reference geometry just created in the previous step. It is not the purpose of this tutorial to describe the (key) abstract concept of `ReferenceFE` in Gridap.

T = Float64
order = 1
pol = Polytope(Fill(HEX_AXIS,D)...)
reffe = LagrangianRefFE(T,pol,order)

# Second, we build the test (Vₕ) and trial (Uₕ) global finite element (FE) spaces out of `model` and `reffe`. At this point we also specify the notion of conformity that we are willing to satisfy, i.e., H1-conformity, and the region of the domain in which we want to (strongly) impose Dirichlet boundary conditions, the whole boundary of the box in this case.

Vₕ = FESpace(model,reffe;conformity=:H1,dirichlet_tags="boundary")
u(x) = x[1]            # Analytical solution (for Dirichlet data)
Uₕ = TrialFESpace(Vₕ,u)

# ## The `CellDatum` abstract type and (some of) its subtypes

# We also want to extract the triangulation out of the model and create a numerical quadrature. We use a quadrature rule with a higher number integration points than those strictly needed to integrate a mass matrix exactly, i.e., `4*order`, instead of `2*order` We do so in order to help the reader distinguish the axis used for quadrature points, and the one used for DoFs in multi-dimensional arrays, which contain the result of evaluating fields (or a differential operator acting on these) in a set of quadrature rule evaluation points.
Tₕ = Triangulation(model)
Qₕ = CellQuadrature(Tₕ,4*order)

# Qₕ is an instance of type `CellQuadrature`, a subtype of the `CellDatum` abstract data type.

isa(Qₕ,CellDatum)

subtypes(CellDatum)

# `CellDatum` is the root of one out of three main type hierarchies in Gridap (along with the ones rooted at the abstract types `Map` and `Field`) on which the evaluation of variational methods in finite-dimensional spaces is grounded on. Any developer of Gridap should familiarize with these three hierarchies to some extent. Along this tutorial we will give some insight on the rationale underlying these, with some examples, but more effort in the form of self-research is expected from the reader as well.

# Conceptually, an instance of a `CellDatum` represents a collection of quantities (e.g., points in a reference system, or scalar-, vector- or tensor-valued fields, or arrays made of these), once per each cell of a triangulation. Using the `get_data` generic function one can extract an array with such quantities. For example, in the case of Qₕ, we get an array of quadrature rules for numerical integration.

Qₕ_cell_data = get_data(Qₕ)
#
@test length(Qₕ_cell_data) == num_cells(Tₕ)

# In this case we get the same quadrature rule in all cells (note that the returned array is of type `Fill`). Gridap also supports different quadrature rules to be used in different cells. Exploring such feature is out of scope of the present tutorial.

# Any `CellDatum` has a trait, the so-called `DomainStyle` trait. This information is consumed by `Gridap` in different parts of the code. It specifies whether the quantities in it are either expressed in the reference (`ReferenceDomain`) or the physical (`PhysicalDomain`) domain. We can indeed check the `DomainStyle` of a `CellDatum` using the `DomainStyle` generic function:

DomainStyle(Qₕ) == ReferenceDomain()
#
DomainStyle(Qₕ) == PhysicalDomain()

# If we evaluate the two expressions above, we can see that the `DomainStyle` trait of Qₕ is `ReferenceDomain`. This means that the local FE space in the physical space in which our problem is posed is expressed in terms of the composition of a space in a reference FE in a parametric space (which is being shared by many or all FEs in the physical space) and the inverse of the geometrical map (from the parametric to the physical space).

# In practise, the integration in the physical space is transformed into a numerical integration in the reference space (via a change of variables) using a quadrature. We can exploit this property for `ReferenceDomain` FE spaces to reduce computations, i.e., to avoid applying the geometrical map to the quadrature points within Qₕ and its inverse at the shape functions in the physical space.

# We note that, while finite elements may not be defined in this parametric space (it is though standard practice with Lagrangian FEs, and other FEs, because of performance reasons), finite element functions are always integrated in such a parametric space. However, for FE spaces that are genuinely defined in the physical space, i.e., the ones with the `PhysicalDomain` trait, the transformation of quadrature points from the reference to the physical space is required.

# In fact, the `DomainStyle` metadata of `CellDatum` allows `Gridap` to do the right thing (as soon as it is implemented) for all combinations of points and FE spaces (both either expressed in the reference or physical space). This is accomplished by the `change_domain` function in the API of `CellDatum`.

# Using the array of quadrature rules `Qₕ_cell_data`, we can access specific entries. The object retrieved provides an array of points (`Point` data type in `Gridap`) in the cell reference parametric space $[0,1]^d$ and their corresponding weights.

q = Qₕ_cell_data[rand(1:num_cells(Tₕ))]
#
p = get_coordinates(q)
#
w = get_weights(q)

# However, there is a more convenient way (for reasons made clear above) to work with the evaluation points of quadratures rules in `Gridap`. Namely, using the `get_cell_points` function we can extract a `CellPoint` object out of a `CellQuadrature`.

Qₕ_cell_point = get_cell_points(Qₕ)

# `CellPoint` (just as `CellQuadrature`) is a subtype of `CellDatum` as well

@test isa(Qₕ_cell_point, CellDatum)

# and thus we can ask for the value of its `DomainStyle` trait, and get an array of quantities out of it using the `get_data` generic function

@test DomainStyle(Qₕ_cell_point) == ReferenceDomain()
#
qₖ = get_data(Qₕ_cell_point)

# Not surprisingly, the `DomainStyle` trait of the `CellPoint` object is `ReferenceDomain`, and we get a (cell) array with an array of `Point`s per each cell out of a `CellPoint`. As seen in the sequel, `CellPoint`s are relevant objects because they are the ones that one can use in order to evaluate the so-called `CellField` objects on the set of points of a `CellPoint`.

# `CellField` is an abstract type rooted at a hierarchy that plays a cornerstone role in the implementation of the finite element method in `Gridap`. At this point, the reader should keep in mind that the finite element method works with global spaces of functions which are defined piece-wise on each cell of the triangulation. In a nutshell (more in the sections below), a `CellField`, as it being a subtype of `CellDatum`, might be understood as a collection of `Field`s (or arrays made out them) per each triangulation cell. `Field` represents a [field](https://simple.wikipedia.org/wiki/Field_(physics)), e.g., a scalar, vector, or tensor field. Thus, the domain of a `Field` are points in the physical domain (represented by a type `Point` in `Gridap`, which is a `VectorValue` with a dimension matching that of the environment space) and the range is a scalar, vector (represented by `VectorValue`) or tensor (represented by `TensorValue`).

# Unlike a plain array of `Field`s, a `CellField` is associated to a triangulation and is specifically designed having in mind FEs. For example, a global finite element function, or the collection of shape basis functions in the local FE space of each cell are examples of `CellField` objects. As commented above, these fields can be defined in the physical or a reference space (combined with a geometrical map provided by the triangulation object for each cell). Thus, `CellField` (as a sub-type of `CellDatum`) has the `DomainStyle` metadata that is used, e.g., for point-wise evaluations (as indicated above) of the fields and their derivatives (by implementing the transformations when taking a differential operators, e.g., the pull-back of the gradients).

# ## Exploring our first `CellField` objects

# Let us work with our first `CellField` objects, namely `FEBasis` objects, and its evaluation. In particular, let us extract out of the global test space, Vₕ, and trial space, Uₕ, a collection of local test and trial finite element shape basis functions, respectively.

dv = get_fe_basis(Vₕ)
#
du = get_trial_fe_basis(Uₕ)

# The objects returned are of `FEBasis` type, one of the subtypes of `CellField`. Apart from `DomainStyle`, `FEBasis` objects also have an additional trait, `BasisStyle`, which specifies whether the cell-local shape basis functions are either of test or trial type (in the Galerkin method). This information is consumed in different parts of the code.

@test Gridap.FESpaces.BasisStyle(dv) == Gridap.FESpaces.TestBasis()
#
@test Gridap.FESpaces.BasisStyle(du) == Gridap.FESpaces.TrialBasis()

# As expected, `dv` is made out of test shape functions, and `du`, of trial shape functions. We can also confirm that both `dv` and `du` are `CellField` and `CellDatum` objects (i.e., recall that `FEBasis` is a subtype of `CellField`, and the latter is a subtype of `CellDatum`).

@test isa(dv,CellField) && isa(dv,CellDatum)
#
@test isa(du,CellField) && isa(du,CellDatum)

# Thus, one may check the value of their `DomainStyle` trait.

@test DomainStyle(dv) == ReferenceDomain()
#
@test DomainStyle(du) == ReferenceDomain()

# We can see that the `DomainStyle` of both `FEBasis` objects is `ReferenceDomain`. In the case of `CellField` objects, this specifies that the point coordinates on which we evaluate the cell-local shape basis functions should be provided in the parametric space of the reference cell (to avoid the need to use the inverse of the geometrical map). However, the output from evaluation, as usual in finite elements defined parametrically, is the cell-local shape function in the physical domain evaluated at the corresponding mapped point.

# Recall from above that `CellField` objects are designed to be evaluated at `CellPoint` objects, and that we extracted a `CellPoint` object, `Qₕ_cell_point`, out of a `CellQuadrature`, of `ReferenceDomain` trait `DomainStyle`. Thus, we can evaluate `dv` and `du` at the quadrature rule evaluation points, on all cells, straight away as:

dv_at_Qₕ = evaluate(dv,Qₕ_cell_point)
#
du_at_Qₕ = evaluate(du,Qₕ_cell_point)

# There are a pair of worth noting observations on the result of the previous two instructions. First, both `dv_at_Qₕ` and `du_at_Qₕ` are arrays of type `Fill` (i.e., a constant array that only stores the entry once) because we are using the same quadrature and reference FE for all cells. This (same entry) is justified by: (1) the local shape functions are evaluated at the same set of points in the reference cell parametric space for all cells (i.e., the quadrature rule points), and (2) the shape functions in physical space have these very same values at the corresponding mapped points in the physical space for all cells. Thus they provide the same entry for whatever index we provide.

dv_at_Qₕ[rand(1:num_cells(Tₕ))]
#
du_at_Qₕ[rand(1:num_cells(Tₕ))]

# At this point, the reader may want to observe which object results from the evaluation of, e.g., `dv_at_Qₕ`, at a different set points for each cell (e.g. by building its own array of arrays of `Points`).

# Going back to our example, any entry of `dv_at_Qₕ` is a rank-2 array of size 9x4 that provides in position `[i,j]` the j-th test shape function at the i-th quadrature rule evaluation point. On the other hand, any entry of `du_at_Qₕ` is a rank-3 array of size `9x1x4` that provides in position `[i,1,j]` the j-th trial shape function at the i-th quadrature point. The reader might be wondering why the rank of these two arrays are different. The rationale is that, by means of the Julia [broadcasting](https://docs.julialang.org/en/v1/manual/arrays/#Broadcasting) of the `*` operation on these two arrays, we get the 9x4x4 array where the `[i,j,k]` entry stores the product of the j-th test and k-th trial functions, both evaluated at the i-th quadrature point. If we sum over the $i$-index, we obtain part of the data required to compute the cell-local matrix that we assemble into the global matrix in order to get a mass matrix. For those readers more used to traditional finite element codes, the broadcast followed by the sum over i, provides the data required in order to implement the following triple standard for-nested loop:

# ```
#  M[:,:]=0.0
#  Loop over quadrature points i
#    detJK_wi=det(JK)*w[i]
#    Loop over shape test functions j
#      Loop over shape trial functions k
#         M[j,k]+=shape_test[i,j]*shape_trial[i,k]*detJK_wi
# ```

# where `det(JK)` represents the determinant of the reference-physical mapping of the current cell, and `w[i]` the quadrature rule weight corresponding to the i-th evaluation point. Using Julia built-in support for broadcasting, we can vectorize the full operation, and get much higher performance.

# The highest-level possible way of performing the aforementioned broadcasted `*` is by building a "new" `CellField` instance by multiplying the two `FEBasis` objects, and then evaluating the resulting object at the points in `Qₕ_cell_point`. This is something common in `Gridap`. One can create new `CellField` objects out of existing ones, e.g., by performing operations among them, or by applying a differential operator, such as the gradient.

dv_mult_du = du*dv
#
dv_mult_du_at_Qₕ = evaluate(dv_mult_du,Qₕ_cell_point)

# We can check that any entry of the resulting `Fill` array is the `9x4x4` array resulting from the broadcasted `*` of the two aforementioned arrays. In order to do so, we can use the so-called `Broadcasting(*)` `Gridap` `Map` (one of the cornerstones of `Gridap`).

# A `Map` represents a (general) function (a.k.a. map or mapping) that takes elements in its domain and return elements in its range. A `Field` is a sub-type of `Map` for the particular domain and ranges of physical fields detailed above. Why do we need to define the `Map` type in `Gridap` instead of using the Julia `Function`? `Map` is essential for performance, as we will explain later on.

# The `Map` below is a map that broadcasts the `*` operation. When applied to arrays of numbers, it essentially translates into the built-in Julia broadcast (check that below!). However, as we will see along the tutorial, such a `Map` can also be applied to, e.g., (cell) arrays of `Field`s (arrays of `Field`s, resp.) to build new (cell) arrays of `Fields` (arrays of `Field`s, resp.). This becomes extremely useful to build and evaluate discrete variational forms.

m=Broadcasting(*)
#
A=evaluate(m,dv_at_Qₕ[rand(1:num_cells(Tₕ))],du_at_Qₕ[rand(1:num_cells(Tₕ))])
#
B=broadcast(*,dv_at_Qₕ[rand(1:num_cells(Tₕ))],du_at_Qₕ[rand(1:num_cells(Tₕ))])
#
@test all(A .≈ B)
#
@test all(A .≈ dv_mult_du_at_Qₕ[rand(1:num_cells(Tₕ))])

# Recall from above that `CellField` objects are also `CellDatum` objects. Thus, one can use the `get_data` generic function to extract, in an array, the collection of quantities, one per each cell of the triangulation, out of them. As one may expect, in the case of our `FEBasis` objects `dv` and `du` at hand, `get_data` returns a (cell) array of arrays of `Field` objects, i.e., the cell-local shape basis functions:

dv_array = get_data(dv)
#
du_array = get_data(du)
#
@test isa(dv_array,AbstractVector{<:AbstractVector{<:Field}})
#
@test isa(du_array,AbstractVector{<:AbstractArray{<:Field,2}})
#
@test length(dv_array) == num_cells(Tₕ)
#
@test length(du_array) == num_cells(Tₕ)

# As expected, both `dv_array` and `du_array` are (*conceptually*) vectors (i.e, rank-1 arrays) with as many entries as cells. The concrete type of each vector differs, though, i.e., `Fill` and `LazyArray`, resp. (We will come back to `LazyArray`s below, as they play a fundamental role in the way in which the finite element method is implemented in `Gridap`.) For each cell, we have arrays of `Field` objects. Recall from above that `Map` and `Field` (with `Field` a subtype of `Map`), and `CellDatum` and `CellField` (with `CellField` a subtype of `CellDatum`) and the associated type hierarchies, are fundamental in `Gridap` for the implementation of variational methods in finite-dimensional spaces. `Field` conceptually represents a physical (scalar, vector, or tensor) field. `Field` objects can be evaluated at single `Point` objects (or at an array of them in one shot), and they return scalars (i.e., a sub-type of Julia `Number`), `VectorValue`, or `TensorValue` objects (or an array of them, resp.)

# In order to evaluate a `Field` object at a `Point` object, or at an array of `Points`, we can use the `evaluate` generic function in its `API`. For example, the following statement

ϕ₃ = dv_array[1][3]
evaluate(ϕ₃,[Point(0,0),Point(1,0),Point(0,1),Point(1,1)])

# evaluates the 3rd test shape function of the local space of the first cell at the 4 vertices of the cell (recall from above that, for the implementation of Lagrangian finite elements being used in this tutorial, shape functions are thought to be evaluated at point coordinates expressed in the parametric space of the reference cell). As expected, ϕ₃ evaluates to one at the 3rd vertex of the cell, and to zero at the rest of vertices, as ϕ₃ is the shape function associated to the Lagrangian node/ DOF located at the 3rd vertex. We can also evaluate all shape functions of the local space of the first cell (i.e., an array of `Field`s) at once at an array of `Points`

ϕ = dv_array[1]
evaluate(ϕ,[Point(0,0),Point(1,0),Point(0,1),Point(1,1)])

# As expected, we get the Identity matrix, as the shape functions of the local space have, by definition, the Kronecker delta property.

# However, and here comes one of the main take-aways of this tutorial, in `Gridap`, (cell-wise) arrays of `Fields` (or arrays of `Fields`) are definitely NOT conceived to be evaluated following the approach that we used in the previous examples, i.e., by manually extracting the `Field` (array of `Field`s) corresponding to a cell, and then evaluating it (them) at a given set of `Point`s. Instead, one uses the `lazy_map` generic function, which combined with the `evaluate` function, represents the operation of walking over all cells, and evaluating the fields, cell by cell, as a whole. This is illustrated in the following piece of code:

dv_array_at_qₖ = lazy_map(evaluate,dv_array,qₖ)
#
du_array_at_qₖ = lazy_map(evaluate,du_array,qₖ)

# We note that the results of these two expressions are equivalent to the ones of `evaluate(dv, Qₕ_cell_point)` and `evaluate(du,Qₕ_cell_point)`, resp. (check it!) In fact, these latter two expressions translate under the hood into the calls to `lazy_map` above. These calls to `lazy_map` return an array of the same length of the input arrays, with their i-th entry conceptually defined, e.g., as `evaluate(du_array[i],qₖ[i])` in the case of the second array. To be "conceptually defined as" does not mean that they are actually computed as `evaluate(du_array[i],qₖ[i])`. Indeed they don't, this would not be high performant.

# You might now be wondering what the main point behind `lazy_map` is. `lazy_map` turns out to be a cornerstone in `Gridap`. (At this point, you may execute `methods(lazy_map)` to observe that a large amount of programming logic is devoted to it.) Let us try to answer it more abstractly now. However, this will be revisited along the tutorial with additional examples.

# `lazy_map` can be applied to a `Map` and an array or a set of arrays, all with the same layout, that provide at every entry the arguments of the map. It conceptually returns the array that results from applying the `Map` to the arguments in each index of the argument array(s). Usually, the resulting type is a `LazyArray`.

# When the resulting `LazyArray` entries are also `Map`s, one could `evaluate` the `LazyArray` on the array(s) that provide the argument(s) (i.e., its domain) using again a `lazy_map`. E.g., for the sub-type `Field`, one can create an array of fields, e.g., cell shape function, apply a `Map` over this array, e.g., a scaling of the shape functions, using `lazy_map`. The resulting array (conceptually also an array of `Field`s) can be evaluated in a set of points applying `evaluate` using `lazy_map`, as in the two code lines above.

# These lazy objects are cornerstones of `Gridap` for the following reasons:

# 1. To keep memory allocation (and consumption) at very low levels, `lazy_map` NEVER returns an array that stores the result at all cells at once. In the two examples above, this is achieved using `Fill` arrays. However, this is only possible in very particular scenarios (see discussion above). In more general cases, the array resulting from `lazy_map` does not have the same entry in all cells. In such cases, `lazy_map` returns a `LazyArray`, which is another essential component of `Gridap`. In a nutshell, a `LazyArray` is an array that applies entry-wise arrays of `Map`s (functions, operations) over array(s) that provide the `Map`s arguments. These operations are only computed when accessing the corresponding index, thus the name lazy. Besides, the entries of these are computed in an efficient way, using a set of mechanisms that will be illustrated below with examples (e.g., using cache to store the entry-wise data without the need to allocate memory each time we access the `LazyArray`).

# 2. Apart from `Function` objects, such as `evaluate`, `lazy_map` can also be used to apply `Map`s to arguments. For example, `Broadcasting(*)` presented above. A `Map` (or its sub-type `Field`) can be applied via `lazy_map` to other `Map`s (or arrays of `Map`s) to build a new `Map` (or array of `Map`s). Thus, the recursive application of `lazy_map` lets us build complex operation trees among arrays of `Map`s as the ones required for the implementation of variational forms. While building these trees, by virtue of Julia support for multiple type dispatching, there are plenty of opportunities for optimization by changing the order in which the operations are performed. These optimizations typically come in the form of a significant saving of FLOPs by exploiting the particular properties of the `Map`s at hand, but could also come from higher granularity for vectorized array operations when the expressions are actually evaluated. Indeed, the arrays that one usually obtains from `lazy_map` differ in some cases from the trivial `LazyArray`s that one would expect from a naive combination of the arguments to `lazy_map`

# 3. Using `lazy_map` we are hiding thousands of cell loops across the code (as the one for the computation of the element matrices above). As a result, `Gridap` is much more expressive for cell-wise implementations.

# ## Exploring another type of `CellField` objects

# Let us now work with another type of `CellField` objects, the ones that are used to represent an arbitrary element of a global FE space of functions, i.e., a FE function. A global FE function can be understood conceptually as a collection of `Field`s, one per each cell of the triangulation. The `Field` corresponding to a cell represents the restriction of the global FE function to the cell. (Recall that in finite elements, global functions are defined piece-wise on each cell.) As we did in the previous section, we will explore, at different levels, how FE functions are evaluated. However, we will dig deeper into this by illustrating some of the aforementioned mechanisms on which `LazyArray` relies in order to efficiently implement the entry-wise application of an operation (or array of operations) to a set of input arrays.

# Let us now build a FE function belonging to the global trial space of functions Uₕ, with [rand](https://docs.julialang.org/en/v1/stdlib/Random/#Base.rand) free DOF values. Using `Gridap` higher-level API, this can be achieved as follows

uₕ = FEFunction(Uₕ,rand(num_free_dofs(Uₕ)))

# As expected from the discussion above, the returned object is a `CellField` object:

@test isa(uₕ,CellField)

# Thus, we can, e.g., query the value of its `DomainStyle` trait, that turns out to be `ReferenceDomain`

@test DomainStyle(uₕ) == ReferenceDomain()

# Thus, in order to evaluate the `Field` object that represents the restriction of the FE function to a given cell, we have to provide `Point`s in the parametric space of the reference cell, and we get the value of the FE function at the corresponding mapped `Point`s in the physical domain. This should not come as a surprise as we have that: (1) the restriction of the FE function to a given cell is mathematically defined as a linear combination of the local shape functions of the cell (with coefficients given by the values of the DOFs at the cell). (2) As observed in the previous section, the shape functions are such that their value at `Point`s that are mapped from the reference cell to the physical cell by the cell geometrical map can be simply be obtained by evaluating the corresponding shape function in the reference FE at the same `Point`s in the parametric space (without the need to compute the geometrical map and its inverse, i.e., exploiting the fact that the combination of this two is the identity map). This property is thus transferred to the FE function.

# As FE functions are `CellField` objects, we can evaluate them at `CellPoint` objects. Let us do it at the points within `Qₕ_cell_point` (see above for a justification of why this is possible):

uₕ_at_Qₕ = evaluate(uₕ,Qₕ_cell_point)

# We note that internally this is just the application of `evaluate` via `lazy_map` for the raw (i.e., without the `CellDatum` metadata) cell arrays of `uₕ` and `Qₕ_cell_point` (obtained via `get_data`). Internally, a `change_domain` is invoked if required, i.e., the two `CellDatum` do not have the same `DomainStyle` trait value (not the case here). You can check it by getting into this call using the VSCode debugger (write `@enter` at the beginning of the line and run it). In any case, we provide many more details below.

# For the first time in this tutorial, we have obtained a cell array of type `LazyArray` from evaluating a `CellField` at a `CellPoint`.

@test isa(uₕ_at_Qₕ,LazyArray)

# This makes sense as a finite element function restricted to a cell is, in general, different in each cell, i.e., it evaluates to different values at the quadrature rule evaluation points. In other words, the `Fill` array optimization that was performed for the evaluation of the cell-wise local shape functions `dv` and `du` does not apply here, and a `LazyArray` has to be used instead.

# Although it is hard to understand the full concrete type name of `uₕ_at_Qₕ` at this time

print(typeof(uₕ_at_Qₕ))

# we will dissect `LazyArray`s in this section up to an extent that will allow us to have a better grasp of it. By now, the most important thing for you to keep in mind is that `LazyArray`s objects encode a recipe to produce its entries just-in-time when they are accessed. They NEVER store all of its entries at once. Even if the expression `uₕ_at_Qₕ` typed in the REPL, (or inline evaluations in your code editor) show the array with all of its entries at once, don't get confused. This is because the Julia REPL is evaluating the array at all indices and collecting the result just for printing purposes.

# Just as we did with `FEBasis`, we can extract an array of `Field` objects out of `uₕ_at_Qₕ`, as `uₕ_at_Qₕ` is also a `CellBasis` object.

uₕ_array = get_data(uₕ)

# As expected, `uₕ_array` is (conceptually) a vector (i.e., rank-1 array) of `Field` objects.

@test isa(uₕ_array,AbstractVector{<:Field})

# Its concrete type is, though, `LazyArray`

@test isa(uₕ_array,Gridap.Fields.LazyArray)

# with full name, as above, of a certain complexity (to say the least):

print(typeof(uₕ_array))

# As mentioned above, `lazy_map` returns `LazyArray`s in the most general scenarios. Thus, it is reasonable to think that `get_data(uₕ)` returns an array that has been built via `lazy_map`. (We advance now that this is indeed the case.) On the other hand, as `uₕ_array` is (conceptually) a vector (i.e., rank-1 array) of `Field` objects, this also tells us that the `lazy_map`/`LazyArray` pair does not only play a fundamental role in the evaluation of (e.g., cell) arrays of `Field`s on (e.g., cell) arrays of arrays of `Point`s, but also in building new cell arrays of `Field`s (i.e., the local restriction of a FE function to each cell) out of existing ones (i.e., the cell array with the local shape functions). In the words of the previous section, we can use `lazy_map` to build complex operation trees among arrays of `Field`s, as required by the computer implementation of variational methods.

# The key question now is: what is the point behind `get_data(uₕ)` returning a `LazyArray` of `Field`s, and not just a plain Julia array of `Field`s? At the end of the day, `Field` objects themselves have very low memory demands, they only need to hold the necessary information to encode their action (evaluation) on a `Point`/array of `Point`s. This is in contrast to the evaluation of (e.g., cell) arrays of `Field`s (or arrays of `Field`s) at an array of `Point`s, which does consume a significantly allocation of memory (if all entries are to be stored at once in memory, and not by demand). The short answer is higher performance. Using `LazyArray`s to encode operation trees among cell arrays of `Field`s, we can apply optimizations when evaluating these operation trees that would not be possible if we just computed a plain array of `Field`s. If all this sounds quite abstract, (most probably it does), we are going to dig into this further in the rest of the section.

# As mentioned above, `uₕ_array` can be conceptually seen as an array of `Field`s. Thus, if we access to a particular entry of it, we should get a `Field` object. (Although possible, this is not the way in which `uₕ_array` is conceived to be used, as was also mentioned in the previous section.) This is indeed confirmed when accessing, e.g., the third entry of `uₕ_array`:

uₕ³ = uₕ_array[3]
#
@test isa(uₕ³,Field)

# The concrete type of `uₕ³` is `LinearCombinationField`. This type represents a `Field` defined as a linear combination of an existing vector of `Field`s. This sort of `Field`s can be built using the `linear_combination` generic function. Among its methods, there is one which takes (1) a vector of scalars (i.e., Julia `Number`s) with the coefficients of the expansion and (2) a vector of `Field`s as its two arguments, and returns a `LinearCombinationField` object. As mentioned above, this is the exact mathematical definition of a FE function restricted to a cell.

# Let us manually build uₕ³. In order to do so, we can first use the `get_cell_dof_values` generic function, which extracts out of uₕ a cell array of arrays with the DOF values of uₕ restricted to all cells of the triangulation (defined from a conceptual point of view).

Uₖ = get_cell_dof_values(uₕ)

# (The returned array turns to be of concrete type `LazyArray`, again to keep memory allocation low, but let us skip this detail for the moment.) If we restrict `Uₖ` and `dv_array` to the third cell

Uₖ³ = Uₖ[3]
#
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

@test all( uₕ_at_Qₕ .≈ manual_uₕ_array_at_qₖ )

# However, and here it comes the key of the discussion, the concrete types of `uₕ_at_Qₕ` and `manual_uₕ_array_at_qₖ` do not match.

@test typeof(uₕ_at_Qₕ) != typeof(manual_uₕ_array_at_qₖ)

# This is because `evaluate(uₕ,Qₕ_cell_point)` does not follow the (naive) approach that we followed to build `manual_uₕ_array_at_qₖ`, but it instead calls `lazy_map` under the hood as follows

uₕ_array_at_qₖ = lazy_map(evaluate,uₕ_array,qₖ)

# Now we can see that the types of `uₕ_array_at_qₖ` and `uₕ_at_Qₕ` match:

@test typeof(uₕ_array_at_qₖ) == typeof(uₕ_at_Qₕ)

# Therefore, why `Gridap` does not build `manual_uₕ_array_at_qₖ`? what's wrong with it? Let us first try to answer this quantitatively. Let us assume that we want to sum all entries of a `LazyArray`. In the case of `LazyArray`s of arrays, this operation is only well-defined if the size of the arrays of all entries matches. This is the case of the `uₕ_array_at_qₖ` and `manual_uₕ_array_at_qₖ` arrays, as we have the same quadrature rule at all cells. We can write this function following the `Gridap` internals' way.

function smart_sum(a::LazyArray)
  cache=array_cache(a)             #Create cache out of a
  sum=copy(getindex!(cache,a,1))   #We have to copy the output
                                   #from get_index! to avoid array aliasing
  for i in 2:length(a)
    ai = getindex!(cache,a,i)      #Compute the i-th entry of a
                                   #re-using work arrays in cache
    sum .= sum .+ ai
  end
  sum
end

# The function uses the so-called "cache" of a `LazyArray`. In a nutshell, this cache can be thought as a place-holder of work arrays that can be re-used among different evaluations of the entries of the `LazyArray` (e.g., the work array in which the result of the computation of an entry of the array is stored.) This way, the code is more performant, as the cache avoids that these work arrays are created repeatedly when traversing the `LazyArray` and computing its entries. It turns out that `LazyArray`s are not the only objects in `Gridap` that (can) work with caches. `Map` and `Field` objects also provide caches for reusing temporary storage among their repeated evaluation on different arguments of the same types. (For the eager reader, the cache can be obtained out of a `Map`/`Field` with the `return_cache` abstract method; see also `return_type`, `return_value`, and `evaluate!` functions of the abstract API of `Map`s). When a `LazyArray` is created out of objects that in turn rely on caches (e.g., a `LazyArray` with entries defined as the entry-wise application of a `Map` to two `LazyArrays`), the caches of the latter objects are also handled by the former object, so that this scheme naturally accommodates top-down recursion, as per-required in the evaluation of complex operation trees among arrays of `Field`s, and their evaluation at a set of `Point`s. We warn the reader this is a quite complex mechanism. The reader is encouraged to follow with a debugger, step by step, the execution of the `smart_sum` function with the `LazyArray`s built above in order to gain some familiarity with this mechanism.

# If we @time the `smart_sum` function with `uₕ_array_at_qₖ` and `manual_uₕ_array_at_qₖ`

smart_sum(uₕ_array_at_qₖ)        # Execute once before to neglect JIT-compilation time
smart_sum(manual_uₕ_array_at_qₖ) # Execute once before to neglect JIT-compilation time
#
@time begin
        for i in 1:100_000
         smart_sum(uₕ_array_at_qₖ)
        end
      end
#
@time begin
        for i in 1:100_000
          smart_sum(manual_uₕ_array_at_qₖ)
        end
      end

# we can observe that the array returned by `Gridap` can be summed in significantly less time, using significantly less allocations.

# Let us try to answer the question now qualitatively. In order to do so, we can take a look at the structure of both `LazyArray`s using the `print_op_tree` function provided by `Gridap`

print_op_tree(uₕ_array_at_qₖ)
#
print_op_tree(manual_uₕ_array_at_qₖ)

# We can observe from the output of these calls the following:

# 1. `uₕ_array_at_qₖ` is a `LazyArray` whose entries are defined as the result of applying a `Fill` array of `LinearCombinationMap{Colon}` `Map`s to a `LazyArray` and a `Fill` array. The first array provides the FE function DOF values restricted to each cell, and the second the local basis shape functions evaluated at the quadrature points. As the shape functions in physical space have the same values in all cells at the corresponding mapped points in physical space, there is no need to re-evaluate them at each cell, we can evaluate them only once. And this is what the second `Fill` array stores as its unique entry, i.e., a matrix `M[i,j]` defined as the value of the j-th `Field` (i.e., shape function) evaluated at the i-th `Point`. *This is indeed the main optimization that `lazy_map` applies compared to our manual construction of `uₕ_array_at_qₖ`.* It is worth noting that, if `v` denotes the linear combination coefficients, and `M` the matrix resulting from the evaluation of an array of `Fields` at a set of `Points`, with `M[i,j]` being the value of the j-th `Field` evaluated at the i-th point, the evaluation of `LinearCombinationMap{Colon}` at `v` and `M` returns a vector `w` with `w[i]` defined as `w[i]=sum_k v[k]*M[i,k]`, i.e., the FE function evaluated at the i-th point. `uₕ_array_at_qₖ` handles the cache of `LinearCombinationMap{Colon}` (which holds internal storage for `w`) and that of the first `LazyArray`, so that when it retrieves the DOF values `v` of a given cell, and then applies `LinearCombinationMap{Colon}` to `v` and `M`, it does not have to allocate any temporary working arrays, but re-uses the ones stored in the different caches.

# 2. `manual_uₕ_array_at_qₖ` is also a `LazyArray`, but structured rather differently to `uₕ_array_at_qₖ`. In particular, its entries are defined as the result of applying a plain array of `LinearCombinationField`s to a `Fill` array of `Point`s that holds the coordinates of the quadrature rule evaluation points in the parametric space of the reference cell (which are equivalent for all cells, thus the `Fill` array). The evaluation of a `LinearCombinationField` on a set of `Point`s ultimately depends on `LinearCombinationMap`. As seen in the previous point, the evaluation of this `Map` requires a vector `v` and a matrix `M`. `v` was built in-situ when building each `LinearCombinationField`, and stored within these instances. However, in contrast to `uₕ_array_at_qₖ`, `M` is not part of `manual_uₕ_array_at_qₖ`, and thus it has to be (re-)computed each time that we evaluate a new `LinearCombinationField` instance on a set of points. This is the main source of difference on the computation times observed. By eagerly constructing our array of `LinearCombinationField`s instead of deferring it until (lazy) evaluation via `lazy_map`, we lost optimization opportunities. We stress that `manual_uₕ_array_at_qₖ` also handles the cache of `LinearCombinationField` (that in turn handles the one of `LinearCombinationMap`), so that we do not need to allocate `M` at each cell, we re-use the space within the cache of `LinearCombinationField`.

# To conclude the section, we expect the reader to be convinced of the negative consequences in performance that an eager (early) evaluation of the entries of the array returned by a `lazy_map` call can have in performance. The leitmotif of `Gridap` is *laziness*. When building new arrays of `Field`s (or arrays of `Field`s), out of existing ones, or when evaluating them at a set of `Point`s, ALWAYS use `lazy_map`. This may expand across several recursion levels when building complex operation trees among arrays of `Field`s. The more we defer the actual computation of the entries of `LazyArray`s, the more optimizations will be available at the `Gridap`'s disposal by re-arranging the order of operations via exploitation of the particular properties of the arrays at hand. And this is indeed what we are going to do in the rest of the tutorial, namely calling `lazy_map` to build new cell arrays out of existing ones, to end in a lazy cell array whose entries are the cell matrices and cell vectors contributions to the global linear system.

# Let us, e.g., build Uₖ manually using this idea. First, we extract out of uₕ and Uₕ two arrays with the free and fixed (due to strong Dirichlet boundary conditions) DOF values of uₕ

uₕ_free_dof_values = get_free_dof_values(uₕ)
#
uₕ_dirichlet_dof_values = get_dirichlet_dof_values(Uₕ)

# So far these are plain arrays, nothing is lazy. Then we extract out of Uₕ the global indices of the DOFs in each cell, the well-known local-to-global map in FE methods.

σₖ = get_cell_dof_ids(Uₕ)

# Finally, we call lazy_map to build a `LazyArray`, whose entries, when computed, contain the global FE function DOFs restricted to each cell.

m = Broadcasting(PosNegReindex(uₕ_free_dof_values,uₕ_dirichlet_dof_values))
#
manual_Uₖ = lazy_map(m,σₖ)

# `PosNegReindex` is a `Map` that is built out of two vectors. We evaluate it at indices of array entries. When we give it a positive index, it returns the entry of the first vector corresponding to this index, and when we give it a negative index, it returns the entry of the second vector corresponding to the flipped-sign index. We can check this with the following expressions

@test evaluate(PosNegReindex(uₕ_free_dof_values,uₕ_dirichlet_dof_values),3) == uₕ_free_dof_values[3]
#
@test evaluate(PosNegReindex(uₕ_free_dof_values,uₕ_dirichlet_dof_values),-7) == uₕ_dirichlet_dof_values[7]

# The `Broadcasting(op)` `Map` lets us, in this particular example, broadcast the `PosNegReindex(uₕ_free_dof_values,uₕ_dirichlet_dof_values)` `Map` to an array a global DOF ids, to obtain the corresponding cell DOF values. As regular, `Broadcasting(op)` provides a cache with the work array required to store its result. `LazyArray` uses this cache to reduce the number of allocations while computing its entries just-in-time. Please note that in `Gridap` we put negative labels to fixed DOFs and positive to free DOFs in σₖ, thus we use an array that combines σₖ with the two arrays of free and fixed DOF values accessing the right one depending on the index. But everything is lazy, only computed when accessing the array. As mentioned multiple times, laziness is one f the leitmotifs in Gridap, the other being immutability.

# Immutability is a feature that comes from functional programming. An immutable object cannot be modified after created. Since objects cannot change, one does not require to track how they change, i.e., there is no need to design (and understand) state diagrams. A code that strictly sticks to this principle is much more readable. Due to laziness, `Gridap` objects are light-weight, and the (lazy) modification of existing (lazy) objects is highly efficient. You can find this action many times in the code above, in which we use `lazy_map` to perform actions over lazy objects (e.g., `LazyArray` or `Fill` arrays) to create new lazy objects. However, strictly conforming to immutability can be inefficient in some very specific scenarios. `Gridap` departs from immutability in the linear algebra part, since we want to re-use the memory allocation as much as possible for global arrays or symbolic/numeric factorisations in linear solvers.

# ## The geometrical model

# From the triangulation we can also extract the cell map, i.e., the geometrical map that takes points in the parametric space $[0,1]^D$ (the `SEGMENT`, `QUAD`, or `HEX` in 1, 2, or 3D, resp.) and maps it to the cell in the physical space $\Omega$.

ξₖ = get_cell_map(Tₕ)

# We note that this map is just a `LazyArray` of `Field`s. The metadata related to `CellField` is not required here, the cell map can only go from the reference to physical space, and its domain can only be a reference cell. For this reason, it is not a `CellField`.

# The cell map takes at each cell points in the parametric space and returns the mapped points in the physical space. Even though this space does not need a global definition (nothing has to be solved here), it is continuous across interior faces.

# As usual, this cell_map is a `LazyArray`. At each cell, it provides the `Field` that maps `Point`s in the parametric space of the reference cell to `Point`s in physical space.

# The node coordinates can be extracted from the triangulation, returning a global array of `Point`s. You can see that such array is stored using Cartesian indices instead of linear indices. It is more natural for Cartesian meshes.

X = get_node_coordinates(Tₕ)

# You can also extract a cell-wise array that provides the node indices per cell

cell_node_ids = get_cell_node_ids(Tₕ)

# or the cell-wise nodal coordinates, combining the previous two arrays

_Xₖ = get_cell_coordinates(Tₕ)

# ## A low-level definition of the cell map

# Now, let us create the geometrical map almost from scratch, using the concepts that we have learned so far. In this example, we consider that the geometry is represented with a bilinear map, and we thus use a first-order, scalar-valued FE space to represent the nodal coordinate values. To this end, as we did before with the global space of FE functions, we first need to create a Polytope using an array of dimension `D` with the parameter `HEX_AXIS`. Then, this is used to create the scalar first order Lagrangian reference FE.

pol = Polytope(Fill(HEX_AXIS,D)...)
reffe_g = LagrangianRefFE(Float64,pol,1)

# Next, we extract the basis of shape functions out of this Reference FE, which is a set of `Field`s, as many as shape functions. We note that these `Field`s have as domain the parametric space $[0,1]^D$. Thus, they can readily be evaluated for points in the parametric space.

ϕrg = get_shapefuns(reffe_g)

# Now, we create a global cell array that has the same reference FE basis for all cells.

ϕrgₖ = Fill(ϕrg,num_cells(Tₕ))

# Next, we use `lazy_map` to build a `LazyArray` that provides the coordinates of the nodes of each cell in physical space. To this end, we use the `Broadcasting(Reindex(X))` and apply it to `cell_node_ids`.

Xₖ = lazy_map(Broadcasting(Reindex(X)),cell_node_ids)

# `Reindex` is a `Map` that is built out of a single vector, `X` in this case. We evaluate it at indices of array entries, and it just returns the entry of the vector from which it is built corresponding to this index. We can check this with the following expressions:

@test evaluate(Reindex(X),3) == X[3]

# If we combine `Broadcasting` and `Reindex`, then we can evaluate efficiently the `Reindex` `Map` at arrays of node ids, i.e., at each of the entries of `cell_node_ids`. `lazy_map` is used for reasons hopefully clear at this point (low memory consumption, efficient computation of the entries via caches, further opportunities for optimizations when combined with other `lazy_map` calls, etc.).
# Finally, we can check that `Xₖ` is equivalent to the array returned by `Gridap`, i.e, `_Xₖ`

@test Xₖ == _Xₖ == get_cell_coordinates(Tₕ) # check

# Next, we can compute the geometrical map as the linear combination of these shape functions in the parametric space with the node coordinates (at each cell)

ψₖ = lazy_map(linear_combination,Xₖ,ϕrgₖ)

# This is the mathematical definition of the geometrical map in FEs! (see above for a description of the `linear_combination` generic function). As expected, the FE map that we have built manually is equivalent to the one internally built by `Gridap`.

@test lazy_map(evaluate,ψₖ,qₖ) == lazy_map(evaluate,ξₖ,qₖ) # check

# It is good to stress (if it was not fully grasped yet) that `lazy_map(k,a,b)`, with `k` being a callable Julia object, is semantically (conceptually) equivalent to `map(k,a,b)` but, among others, with a lazy result instead of a plain Julia array. A Julia object is callable if it makes sense to pass arguments to it. For example, objects `k` such that `isa(k,Map)==true` are callable.  For these objects, `k(x...)` is equivalent to `evaluate(k,x...)`.

# Following the same ideas, we can compute the Jacobian of the geometrical map (cell-wise). The Jacobian of the transformation is simply its gradient. The gradient in the parametric space can be built using two equivalent approaches. On the one hand, we can apply the `Broadcasting(∇)` `Map` to the array of `Fields` with the local shape basis functions (i.e., `ϕrg`). This results in an array of `Field`s with the gradients, (Recall that `Map`s can be applied to array of `Field`s in order to get new array of `Field`s) that we use to build a `Fill` array with the result. Finally, we build the lazy array with the cell-wise Jacobians of the map as the linear combination of the node coordinates and the gradients of the local cell shape basis functions:

∇ϕrg  = Broadcasting(∇)(ϕrg)
#
∇ϕrgₖ = Fill(∇ϕrg,num_cells(model))
#
J = lazy_map(linear_combination,Xₖ,∇ϕrgₖ)

# We note that `lazy_map` is not required in the first expression, as we are not actually working with cell arrays. On the other hand, using `lazy_map`, we can apply `Broadcasting(∇)` to the cell array of `Field`s with the geometrical map.

lazy_map(Broadcasting(∇),ψₖ)

# As mentioned above, those two approaches are equivalent

@test typeof(J) == typeof(lazy_map(Broadcasting(∇),ψₖ))
#
@test lazy_map(evaluate,J,qₖ) == lazy_map(evaluate,lazy_map(Broadcasting(∇),ψₖ),qₖ)

# ## Computing the gradients of the trial and test FE space bases

# Another salient feature of Gridap is that we can directly take the gradient of finite element bases. (In general, of any `CellField` object.) In the following code snippet, we do so for `dv` and `du`

grad_dv = ∇(dv)
#
grad_du = ∇(du)

# The result of this operation when applied to a `FEBasis` object is a new `FEBasis` object.

@test isa(grad_dv, Gridap.FESpaces.FEBasis)
#
@test isa(grad_du, Gridap.FESpaces.FEBasis)

# We can also extract an array of arrays of `Fields`, as we have done before with `FEBasis` objects.

grad_dv_array = get_data(grad_dv)
#
grad_du_array = get_data(grad_du)

# The resulting `LazyArray`s encode the so-called pull back transformation of the gradients. We need this transformation in order to compute the gradients in physical space. The gradients in physical space are indeed the ones that we need to integrate in the finite element method, not the reference ones, even if we always evaluate the integrals in the parametric space of the reference cell. We can also check that the `DomainStyle` trait of `grad_dv` and `grad_du` is `ReferenceDomain`

@test DomainStyle(grad_dv) == ReferenceDomain()
#
@test DomainStyle(grad_du) == ReferenceDomain()

# This should not come as a surprise, as this is indeed the nature of the pull back transformation of the gradients. We provide `Point`s in the parametric space of the reference cell, and we get back the gradients in physical space evaluated at the mapped `Point`s in physical space.

# We can manually build `grad_dv_array` and `grad_du_array` as follows
ϕr                   = get_shapefuns(reffe)
∇ϕr                  = Broadcasting(∇)(ϕr)
∇ϕrₖ                 = Fill(∇ϕr,num_cells(Tₕ))
manual_grad_dv_array = lazy_map(Broadcasting(push_∇),∇ϕrₖ,ξₖ)
#
∇ϕrᵀ                 = Broadcasting(∇)(transpose(ϕr))
∇ϕrₖᵀ                = Fill(∇ϕrᵀ,num_cells(Tₕ))
manual_grad_du_array = lazy_map(Broadcasting(push_∇),∇ϕrₖᵀ,ξₖ)

# We note the use of the `Broadcasting(push_∇)` `Map` at the last step. For Lagrangian FE spaces, this `Map` represents the pull back of the gradients. This transformation requires the gradients of the shape functions in the reference space, and the (gradient of the) geometrical map. The last step, e.g., the construction of `manual_grad_dv_array`, actually translates into the combination of the following calls to `lazy_map` to build the final transformation:

# Build array of `Field`s with the Jacobian transposed at each cell
Jt     = lazy_map(Broadcasting(∇),ξₖ)

# Build array of `Field`s with the inverse of the Jacobian transposed at each cell
inv_Jt = lazy_map(Operation(inv),Jt)

# Build array of arrays of `Field`s defined as the broadcasted single contraction of the Jacobian inverse transposed and the gradients of the shape functions in the reference space
low_level_manual_gradient_dv_array = lazy_map(Broadcasting(Operation(⋅)),inv_Jt,∇ϕrₖ)

# As always, we check that all arrays built are are equivalent
@test typeof(grad_dv_array) == typeof(manual_grad_dv_array)
#
@test lazy_map(evaluate,grad_dv_array,qₖ) == lazy_map(evaluate,manual_grad_dv_array,qₖ)
#
@test lazy_map(evaluate,grad_dv_array,qₖ) == lazy_map(evaluate,low_level_manual_gradient_dv_array,qₖ)
#
@test lazy_map(evaluate,grad_dv_array,qₖ) == evaluate(grad_dv,Qₕ_cell_point)

# With the lessons learned so far in this section, it is left as an exercise for the reader to manually build the array that `get_data` returns when we call it with the `CellField` object resulting from taking the gradient of uₕ as an argument, i.e., `get_data(∇(uₕ))`.

# ## A low-level implementation of the residual integration and assembly

# Let us now create manually an array of `Field`s uₖ that returns the FE function uₕ at each cell, and another array with its gradients, ∇uₖ. We hope that the next set of instructions can be already understood with the material covered so far

ϕrₖ = Fill(ϕr,num_cells(Tₕ))
#
∇ϕₖ = manual_grad_dv_array
#
uₖ  = lazy_map(linear_combination,Uₖ,ϕrₖ)
#
∇uₖ = lazy_map(linear_combination,Uₖ,∇ϕₖ)

# Let us consider now the integration of (bi)linear forms. The idea is to
# compute first the following residual for our random function uₕ

intg = ∇(uₕ)⋅∇(dv)

# but we are going to do it using low-level methods instead.

# First, we create an array that for each cell returns the dot product of the gradients

Iₖ = lazy_map(Broadcasting(Operation(⋅)),∇uₖ,∇ϕₖ)

# This array is equivalent to the one within the `intg` `CellField` object

@test all(lazy_map(evaluate,Iₖ,qₖ) .≈ lazy_map(evaluate,get_data(intg),qₖ))

# Now, we can finally compute the cell-wise residual array, which using the high-level `integrate` function is

res = integrate(∇(uₕ)⋅∇(dv),Qₕ)

# In a low-level, what we do is to apply (create a `LazyArray`) the `IntegrationMap` `Map` over the integrand evaluated at the integration points, the quadrature rule weights, and the Jacobian evaluated at the integration points

Jq = lazy_map(evaluate,J,qₖ)
intq = lazy_map(evaluate,Iₖ,qₖ)
iwq = lazy_map(IntegrationMap(),intq,Qₕ.cell_weight,Jq)
#
@test all(res .≈ iwq)

# The result is the cell-wise residual (previous to assembly). This is a lazy array but you could collect the element residuals into a plain Julia array if you want

collect(iwq)

# Alternatively, we can use the following syntactic sugar

cellvals = ∫( ∇(dv)⋅∇(uₕ) )*Qₕ

# and check that we get the same cell-wise residual as the one defined above

@test all(cellvals .≈ iwq)

# ## Assembling a residual

# Now, we need to assemble these cell-wise (lazy) residual contributions in a global (non-lazy) array. With all this, we can assemble our vector using the cell-wise residual contributions and the assembler. Let us create a standard assembler struct for the finite element spaces at hand. This will create a vector of size global number of DOFs, and a `SparseMatrixCSC`, to which we can add contributions.

assem = SparseMatrixAssembler(Uₕ,Vₕ)

# We create a tuple with 1-entry arrays with the cell vectors (i.e., `iwq`) and cell-wise global DOF identifiers (i.e., `σₖ`). If we had additional terms, we would have more entries in the array. You can take a look at the `SparseMatrixAssembler` struct for more details.

#
rs = ([iwq],[σₖ])
#
b = allocate_vector(assem,rs)
#
assemble_vector!(b,assem,rs)

# ## A low-level implementation of the Jacobian integration and assembly

# After computing the residual, we use similar ideas for the Jacobian. The process is the same as above, so it does not require additional explanations

∇ϕₖᵀ = manual_grad_du_array
int = lazy_map(Broadcasting(Operation(⋅)),∇ϕₖ,∇ϕₖᵀ)
#
@test all(collect(lazy_map(evaluate,int,qₖ)) .==
            collect(lazy_map(evaluate,get_data(∇(du)⋅∇(dv)),qₖ)))
#
intq = lazy_map(evaluate,int,qₖ)
Jq = lazy_map(evaluate,J,qₖ)
iwq = lazy_map(IntegrationMap(),intq,Qₕ.cell_weight,Jq)
#
jac = integrate(∇(dv)⋅∇(du),Qₕ)
#
@test collect(iwq) == collect(jac)
#
rs = ([iwq],[σₖ],[σₖ])
#
A = allocate_matrix(assem,rs)
#
A = assemble_matrix!(A,assem,rs)

# Now we can obtain the free DOFs and add the solution to the initial guess

x = A \ b
uf = get_free_dof_values(uₕ) - x
ufₕ = FEFunction(Uₕ,uf)
#
@test sum(integrate((u-ufₕ)*(u-ufₕ),Qₕ)) <= 10^-8

# or if you like Unicode symbols

@test ∑(∫(((u-ufₕ)*(u-ufₕ)))Qₕ) <= 10^-8
