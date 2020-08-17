# Disclaimer: This tutorial is about a low-level definition of finite element
# methods. It would be nice to have two (or three) previous tutorials, about
# `Field`, the lazy array machinery related to `AppliedArray` for _numbers_,
# and the combination of both to create lazy arrays that involve fields. This
# is work in progress.

# This tutorial is advanced and you only need to go through this if you want
# to know the internals of `Gridap` and what it is doing under the hood.
# Even though you will likely want to use the high-level APIs in `Gridap`,
# this tutorial will (hopefully) help if you want to become a `Gridap` developer,
# not just a user. We also consider that this tutorial shows how powerful and
# expressive the `Gridap` kernel is, and how mastering it you can implement new
# algorithms not been provided by the library.

# Let us start including `Gridap` and some of its packages, to have access to
# a rich set of not so high-level methods
# Note that the module `Gridap` provides the high-level API, whereas the sub-modules
# like `Gridap.FESpaces` provide access to the different parts of the low-level API.

using Gridap
using Gridap.FESpaces
using Gridap.ReferenceFEs
using Gridap.Arrays
using Gridap.Geometry
using Gridap.Fields
using FillArrays
using Test

# We first create the geometry model and FE spaces using the
# high-level interface. In this tutorial, we are not going to describe the
# most of the geometrical machinery in detail, only what is relevant for the
# discussion. To simplify the analysis of the outputs,
# you can consider a 2D mesh, i.e., `D=2` (everything below works for any dim without
# any extra complication). In order to make things slightly more interesting,
# e.g., having non-constant Jacobians, we have considered a mesh that is
# a stretching of an equal-sized structured mesh.

L = 2 # Domain length
D = 2 # dim
n = 4 # parts per dim

function stretching(x::Point)
m = zeros(length(x)) # zero(mutable(x))
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

# Now, we define the finite element (FE) spaces (Lagrangian, scalar, H1-conforming, i.e.
# continuous). We are going to extract from these FE spaces some information to
# be used in the low-level handling of cell-wise arrays.

u(x) = x[1] # Analytical solution (for Dirichlet data)

T = Float64; order = 1
Vₕ = FESpace(model=model,valuetype=T,reffe=:Lagrangian,order=1,
             conformity=:H1,dirichlet_tags="boundary")
Uₕ = TrialFESpace(Vₕ,u)

# We also want to extract the triangulation of the model and obtain the quadrature.

Tₕ = Triangulation(model)
Qₕ = CellQuadrature(Tₕ,2*order)

# A quadrature provides an array (cells) of arrays of points (`Point` in
# `Gridap`) in the parametric space in which finite element spaces are usually
# defined (and always integrated) and their corresponding weights.

qₖ = get_coordinates(Qₕ)
wₖ = get_weights(Qₕ)

# ## Exploring FE functions in `Gridap`

# A FE function with [rand](https://docs.julialang.org/en/v1/stdlib/Random/#Base.rand)
# free dofs in Uₕ can be defined as follows

uₕ = FEFunction(Uₕ,rand(num_free_dofs(Uₕ)))

# We can extract the array that at each cell of the mesh returns a `Field`,
# the FE function restricted to that cell

uₖ = get_array(uₕ)

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

σₖ = get_cell_dofs(Uₕ)

# Finally, we can extract the vector of values.

Uₖ = get_cell_values(uₕ)

# Take a look at the type of array
# it is. In Gridap we put negative labels to fixed DOFs and positive to free DOFs,
# thus we use an array that combines σₖ with the two arrays of free and fixed values
# accessing the right one depending on the index. But everything is lazy, only
# computed when accessing the array. Laziness and quasi-immutability are leitmotifs in
# Gridap.

# We can also extract an array that provides at each cell the finite element
# basis in the physical space, which are again fields.

dv = get_cell_basis(Vₕ)
du = get_cell_basis(Uₕ)

# We note that these bases differ from the fact that the first one is of
# test type and the second one of trial type (in the Galerkin method). This information
# is consumed in different parts of the code.

is_test(dv) # true

##

is_trial(du) # true

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

ctn = get_cell_nodes(Tₕ)

# or the cell-wise nodal coordinates, combining the previous two arrays

_Xₖ = get_cell_coordinates(Tₕ)

# ## A low-level definition of the cell map

# Now, let us create the geometrical map almost from scratch, in order to
# get familiarised with the `Gridap` internals.
# First, we start with the reference topology of the representation that we
# will use for the geometry. In this example, we consider that the geometry
# is represented with a bilinear map, and we use a scalar-valued FE space to
# combine the nodal coordinate values which is a Lagrangian first order space
# It is not the purpose of this tutorial to describe the `ReferenceFE` in Gridap.

pol = Polytope((Fill(HEX_AXIS,D)...))
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
# lazy array of arrays of `Point`, i.e., the nodes per each cell.

Xₖ = LocalToGlobalArray(ctn,X)

#

@test Xₖ == _Xₖ == get_cell_coordinates(Tₕ) # check

# Even though the `@show` method is probably showing the full matrix, don't get
# confused. This is because this method is evaluating the array at all indices and
# collecting and printing the result. In practical runs, this array, as many other in
# `Gridap`, is lazy. We only compute its entries for a given index on demand, by
# accessing to the pointer array `lcn` and extract the values in `X`.

# Next, we can compute the geometrical map as the combination of these shape
# functions in the parametric space with the node coordinates (at each cell)

lc = Gridap.Fields.LinComValued()
lcₖ = Fill(lc,num_cells(model))
ψₖ = apply(lcₖ,ϕrgₖ,Xₖ)

# Note that since we use the same kernel for all cells, we don't need to build the array of kernels
# `lcₖ`, we can simply write

ψₖ = apply(lc,ϕrgₖ,Xₖ)

#

@test evaluate(ψₖ,qₖ) == evaluate(ξₖ,qₖ) # check

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
# index. This way, the code is performant, and does involve allocations when
# traversing these arrays. It is probably a good time to take a look at `AppliedArray`
# and the abstract API of `Kernel` in `Gridap`.

# It is good to mention that `apply(k,a,b)`` is equivalent to
# map((ai,bi)->apply_kernel(k,ai,bi),a,b) but with a lazy result instead of a
# plain julia array.

# With this, we can compute the Jacobian (cell-wise).
# The Jacobian of the transformation is simply its gradient.
# The gradient in the parametric space can be computed as a gradient of the
# global array defined before, or taking the gradient and filling the array

∇ϕrgₖ = Fill(∇(ϕrg),num_cells(model))

#

@test evaluate(∇ϕrgₖ,qₖ) == evaluate(∇(ϕrgₖ),qₖ)

#

J = apply(lc,∇ϕrgₖ,Xₖ)

#

@test all(evaluate(J,qₖ) .≈ evaluate(∇(ξₖ),qₖ))

# ## A low-level definition of FE space bases

# We proceed as before, creating the reference FE, the reference basis, and the
# corresponding constant array.

pol = Polytope((Fill(HEX_AXIS,D)...))
reffe = LagrangianRefFE(T,pol,order)

ϕr = get_shapefuns(reffe)
ϕrₖ = Fill(ϕr,num_cells(model))

# As stated in FE theory, we can now define the shape functions in the physical
# space, which are conceptually $ϕ(x) = ϕr_K(X)∘ξ_K^{-1}(x)$. We provide a kernel
# for this, `AddMap`. First, we create this kernel,
# and then we apply it to the basis in the parametric space and
# the geometrical map

map = Gridap.Fields.AddMap()
ϕₖ = apply(map,ϕrₖ,ξₖ)

@test ϕₖ === attachmap(ϕrₖ,ξₖ)

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

bₖ = GenericCellBasis(Val{false}(),ϕₖ,ξₖ,Val{true}())

# We can check that the basis we have created return the same values as the
# one obtained with high-level APIs

@test collect(evaluate(dv,qₖ)) == collect(evaluate(bₖ,qₖ))

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

grad = Gridap.Fields.Valued(Gridap.Fields.PhysGrad())
∇ϕrₖ = Fill(Gridap.Fields.FieldGrad(ϕr),num_cells(Tₕ))
∇ϕₖ = apply(grad,∇ϕrₖ,J)

#

@test evaluate(∇ϕₖ,qₖ) == evaluate(∇(ϕₖ),qₖ)

# We can now evaluate both the CellBasis and the array of physical shape functions,
# and check we get the same.

@test evaluate(∇ϕₖ,qₖ) == evaluate(∇(bₖ),qₖ) == evaluate(∇(dv),qₖ)

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

lc = Gridap.Fields.LinComValued()
uₖ = apply(lc,ϕₖ,Uₖ)

# We can check that we get the same results as uₕ

@test evaluate(uₖ,qₖ) == evaluate(uₕ,qₖ)

# Now, since we can apply the gradient over this array

gradient(uₖ)

# or compute it using low-level methods, as a linear combination
# of ∇(ϕₖ) instead of ϕₖ

∇uₖ = apply(lc,∇ϕₖ,Uₖ)
aux = ∇(uₖ)

# We can check we get the expected result

@test evaluate(∇uₖ,qₖ) == evaluate(aux,qₖ)

# ## A low-level implementation of the residual integration and assembly

# We have the array uₖ that returns the finite element function uₕ at
# each cell, and its gradient ∇uₖ.
# Let us consider now the integration of (bi)linear forms. The idea is to
# compute first the following residual for our random function uₕ

intg = ∇(uₕ)⋅∇(dv)

# but we are going to do it using low-level methods.

# First, we create an array that for each cell returns the dot operator

dotop = Gridap.Fields.FieldBinOp(dot)
dotopv = Gridap.Fields.Valued(dotop)
Iₖ = apply(dotopv,∇uₖ,∇ϕₖ)
# Next we consider a lazy `AppliedArray` that applies the `dot_ₖ` array of
# operations (binary operator) over the gradient of the FE function and
# the gradient of the FE basis in the physical space

@test evaluate(intg,qₖ) == evaluate(Iₖ,qₖ)

# Now, we can finally compute the cell-wise residual array, which using
# the high-level `integrate` function is

get_free_values(uₕ)
res = integrate(∇(uₕ)⋅∇(dv),Tₕ,Qₕ)

# In a low-level, what we do is to apply (create a `AppliedArray`)
# the `IntKernel` over the integrand evaluated at the integration
# points, the weights, and the Jacobian evaluated at the integration points

Jq = evaluate(J,qₖ)
intq = evaluate(Iₖ,qₖ)
iwq = apply(Gridap.Fields.IntKernel(),intq,wₖ,Jq)

@test all(res .≈ iwq)

# The result is the cell-wise residual (previous to assembly). This is a lazy
# array but you could collect the element residuals if you want

collect(iwq)

# Alternatively, we could use the high-level API that creates a `LinearFETerm`
# that is the composition of a lambda-function or
# [anonymous function](https://docs.julialang.org/en/v1/manual/functions/#man-anonymous-functions-1)
# with the bilinear form, triangulation and quadrature

blf(u,v) = ∇(u)⋅∇(v)
term = LinearFETerm(blf,Tₕ,Qₕ)

# and check that we get the same residual as the one defined above

cellvals = get_cell_residual(term,uₕ,dv)
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

cellids = get_cell_id(Tₕ) # == identity_vector(num_cells(trian))

rs = ([iwq],[cellids])
b = allocate_vector(assem,rs)
assemble_vector!(b,assem,rs)

# ## A low-level implementation of the Jacobian integration and assembly

# After computing the residual, we use similar ideas for the Jacobian.
# The process is the same as above, so it does not require more explanations

int = apply(dotopv,∇(ϕₖ),∇(ϕₖ))
@test all(collect(evaluate(int,qₖ)) .== collect(evaluate(∇(du)⋅∇(dv),qₖ)))

intq = evaluate(int,qₖ)
Jq = evaluate(J,qₖ)
iwq = apply(Gridap.Fields.IntKernel(),intq,wₖ,Jq)

jac = integrate(int,Tₕ,Qₕ)
@test collect(iwq) == collect(jac)

rs = ([iwq],[cellids],[cellids])
A = allocate_matrix(assem,rs)
A = assemble_matrix!(A,assem,rs)

# Now we can obtain the free dofs and add the solution to the initial guess

x = A \ b
uf = sol = get_free_values(uₕ) - x
ufₕ = FEFunction(Uₕ,uf)

#

@test sum(integrate((u-ufₕ)*(u-ufₕ),Tₕ,Qₕ)) <= 10^-8
