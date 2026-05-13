# In this tutorial, we will learn
#
#  - How the `Map` interface works and how to define custom maps
#  - How `Broadcasting` and `Operation` compose maps lazily
#  - How `lazy_map` defers computation and what type it returns for different inputs
#  - What `Fill` is and how `lazy_map` propagates it without creating a `LazyArray`
#  - How `LazyArray` stores expression trees and how to traverse one with zero allocation
#  - How `CompressedArray` stores data and how `lazy_map` preserves its structure
#  - How `Table` encodes lists-of-lists and the key operations on it
#  - How `Reindex` and `PosNegReindex` provide lazy fancy indexing

using Gridap
using Gridap.Arrays
using FillArrays
using Test

#
# ## The Map interface
#
# A `Map` in Gridap is any callable that participates in the caching and lazy
# evaluation machinery.  The interface has four methods:
#
# | Method | Purpose |
# |--------|---------|
# | `return_cache(f, x...)` | allocate reusable workspace |
# | `evaluate!(cache, f, x...)` | compute, reusing the workspace |
# | `evaluate(f, x...)` | allocate cache + evaluate (one-shot) |
# | `return_type(f, x...)` | infer return type without computing |
#
# Plain Julia functions are Maps by default: `evaluate!(c, f, x...) = f(x...)`
# with `nothing` as the cache.  The cache matters for operations that write into
# pre-allocated buffers — the same workspace is reused across calls, so no
# allocation occurs inside a hot loop.

f = x -> 2x
cache = return_cache(f, 3)
@test isnothing(cache)
@test evaluate!(cache, f, 3) == 6
@test return_type(f, 3) == Int

# `Broadcasting(g)` wraps a scalar function `g` so that it broadcasts
# element-wise over arrays.  Its cache is a pre-allocated output array.

bm = Broadcasting(+)
a  = rand(3, 2)
b  = rand(3, 2)
cache = return_cache(bm, a, b)
c = evaluate!(cache, bm, a, b)
@test c ≈ a .+ b

# Calling `evaluate!` a second time reuses `cache`'s buffer — no new allocation.
# This is how Gridap avoids allocating at every cell during assembly.

c2 = evaluate!(cache, bm, a, b)
@test c2 === c                  # same array object, not a copy

# `Operation(f)(g, h)` composes: it represents `x ↦ f(g(x), h(x))`.

fa(x) = 2 .* x
fb(x) = sqrt.(x)
fab = Operation(fa)(fb)
x = fill(4.0, 3, 3)
@test evaluate(fab, x) ≈ fa(fb(x))

# ## Array interfaces
#
# ### lazy_map
#
# `lazy_map(f, a, b, ...)` is the cell-array analogue of `map`: it applies `f`
# element-wise, but **defers computation**.  The concrete return type is not
# always `LazyArray` — Gridap inspects the input types and applies structural
# optimisations.
# We will see it in action later on in the tutorial.
#
# ### The Gridap Array API
#
# Beyond the standard Julia `AbstractArray` interface (`size`, `getindex`, …),
# Gridap adds three methods that enable the zero-allocation traversal used
# throughout assembly:
#
# | Method | Purpose |
# |--------|---------|
# | `array_cache(a)` | allocate a reusable workspace for element access |
# | `getindex!(cache, a, i)` | fetch element `i`, writing into the cache buffer |
# | `testitem(a)` | return a representative element (used for type inference) |
#
# The default implementations work for any `AbstractArray`, but custom types
# can override them to avoid per-access allocation entirely.
#
# To see this in action we implement `MyScaledArray`: an array whose element
# `i` is `scale * data[i]`, where `data[i]` is itself a vector.  The cache
# pre-allocates one output buffer so that `getindex!` never allocates.

struct MyScaledArray{T} <: AbstractVector{Vector{T}}
  data  :: Vector{Vector{T}}
  scale :: T
end

Base.size(a::MyScaledArray)                  = size(a.data)
Base.IndexStyle(::Type{<:MyScaledArray})     = IndexLinear()
Base.getindex(a::MyScaledArray, i::Integer)  = a.scale .* a.data[i]  # allocates

# The cache is a single pre-allocated output vector.  `getindex!` writes into
# it and returns it — the caller must not hold on to the reference across calls.
# We provide `CachedArray` as a convenient wrapper that handles resizing when needed.
# The cache is reusable across calls to `getindex!`, and can be resized with `setsize!`.

Arrays.array_cache(a::MyScaledArray) = CachedArray(similar(a.data[1]))

function Arrays.getindex!(cache, a::MyScaledArray, i::Integer)
  di = a.data[i]
  setsize!(cache, size(di))   # resize cache if needed
  r = cache.array
  r .= a.scale .* a.data[i]
  return r
end

# `testitem` provides a representative element for type inference when the
# array might be empty. Defaults to `first(a)`.

Arrays.testitem(a::MyScaledArray) = a.scale .* a.data[1]

# Construction and basic access:

raw = [rand(i) for i in 1:8]
msa = MyScaledArray(raw, 3.0)
@test msa[2] ≈ 3.0 .* raw[2]

# Zero-allocation traversal: the same buffer is returned every iteration.

buf = array_cache(msa)
for i in 1:length(msa)
  v = getindex!(buf, msa, i)   # no allocation; v is buf
  @test v ≈ 3.0 .* raw[i]
end

# Because `MyScaledArray` implements the interface, `lazy_map` can use it as
# an argument and will call `array_cache`/`getindex!` on it internally:

doubled = lazy_map(v -> 2 .* v, msa)
@test doubled[3] ≈ 6.0 .* raw[3]

#
# ### Fill arrays and the Fill optimisation
#
# `Fill(v, n)` (from FillArrays.jl) is a length-`n` array whose every element
# is `v`.  It stores `v` exactly once.
#
# When all inputs to `lazy_map` are `Fill`, Gridap evaluates the function
# **once** on the stored value and returns a new `Fill` — no `LazyArray` is
# created.

c = lazy_map(fa, Fill(4.0, 10))
@test isa(c, Fill)
@test c[1] == fa(4.0)
@test c[7] == fa(4.0)

# The same optimisation applies when mixing `Fill` with maps:

maps = Fill(Broadcasting(+), 10)
xs   = Fill(rand(2, 3), 10)
ys   = Fill(rand(2, 3), 10)
v    = lazy_map(evaluate, maps, xs, ys)
@test isa(v, Fill)

# This is why evaluating shape functions at quadrature points on a uniform mesh
# yields a `Fill`: the same reference element and the same quadrature rule are
# reused for every cell, so the result is the same matrix everywhere.

#
# ### LazyArray
#
# When inputs are not uniform, `lazy_map` returns a `LazyArray`.
# A `LazyArray{G,T,N,F}` stores:
# - `maps`: an array of `Map`s (often a `Fill` of a single map)
# - `args`: a tuple of argument arrays
#
# Entries are computed on demand.  Calling `a[i]` allocates a fresh result;
# for hot loops, use the **cache interface** instead.

a = [rand(3) for _ in 1:6]
b = [rand(3) for _ in 1:6]
c = lazy_map(Broadcasting(+), a, b)
d = lazy_map(Broadcasting(*), a, c)

# `d` is the expression tree `a .* (a .+ b)`, unevaluated.

@test d isa LazyArray

# `array_cache` builds the cache for a (possibly nested) `LazyArray`.
# `getindex!` evaluates entry `i` into the cache's buffer — zero allocation
# after the first call.

cache = array_cache(d)
for i in 1:length(d)
  di = getindex!(cache, d, i)   # result lives in cache buffer; no allocation
  @test di ≈ a[i] .* (a[i] .+ b[i])
end

# Calling `getindex!` twice for the same index returns the **same array object**
# without recomputing — the cache memoises the last result.

cache = array_cache(d)
@test getindex!(cache, d, 1) === getindex!(cache, d, 1)

# When `a` appears at multiple nodes of the expression tree (in both the sum
# and the product above), the cache ensures `a[i]` is read only once per index.
# This structural sharing is the main reason cell-local assembly in Gridap is
# efficient: shape functions evaluated at quadrature points are computed once
# and reused across all integration terms.
#
# Multi-dimensional `LazyArray`s support both linear and Cartesian indexing:

s = (3, 4)
x = rand(s...)
y = rand(s...)
z = lazy_map(+, x, y)
@test size(z) == s
@test z[2, 3] == x[2, 3] + y[2, 3]

# `print_op_tree(d)` is a useful debugging utility that prints the full
# expression tree of a `LazyArray`, showing how maps and arguments are composed.

#
# ### CompressedArray
#
# `CompressedArray(values, ptrs)` stores a short `values` vector and an integer
# `ptrs` vector.  Element `i` is `values[ptrs[i]]`.  This arises naturally
# whenever many cells share the same data — for example, on a uniform mesh all
# cells of the same type share the same reference FE.

values = [10, 20, 31]
ptrs   = [1, 2, 3, 3, 2, 2]
a = CompressedArray(values, ptrs)
@test a[1] == 10
@test a[4] == 31   # ptrs[4] == 3 → values[3] == 31
@test a[5] == 20   # ptrs[5] == 2 → values[2] == 20

# When **all** inputs to `lazy_map` are `CompressedArray`s with **identical
# `ptrs`**, Gridap applies the operation only to `values` and returns a new
# `CompressedArray`.  The output has exactly as many unique entries as the
# input, regardless of the array length.

b = lazy_map(-, a)
@test isa(b, CompressedArray)
@test b.ptrs === a.ptrs         # ptrs object is reused
@test b.values == -values

c = lazy_map(*, a, b)
@test isa(c, CompressedArray)
@test c.values == values .* (-values)

# Mixing a `CompressedArray` with a `Fill` also works: the `Fill` is
# conceptually "compressed" to align with the `CompressedArray`'s structure.

f = Fill(4, length(ptrs))
c = lazy_map(*, a, f)
@test isa(c, CompressedArray)
@test c.ptrs === a.ptrs

# Two `CompressedArray`s with **different** `ptrs` cannot share structure —
# Gridap falls back to a `LazyArray`.

values2 = [10, 20]
ptrs2   = [1, 2, 1, 1, 2, 2]
b2 = CompressedArray(values2, ptrs2)
c  = lazy_map(*, a, b2)
@test !isa(c, CompressedArray)
@test  isa(c, LazyArray)

#
# ## Tables
#
# `Table(data, ptrs)` is a compressed list-of-lists.  Row `i` spans
# `data[ptrs[i] : ptrs[i+1]-1]`.  The `Table` type is used throughout Gridap
# for mesh connectivity: cell-to-node maps, cell-to-DOF maps, etc.

vv = [[1, 2, 3], [2, 3], [5, 8], Int[], [1, 2, 4]]
t  = Table(vv)
@test length(t) == 5
@test t[1] == [1, 2, 3]
@test t[4] == Int[]

# `view(t, i)` returns a view of row `i` into the flat `data` array (no copy).
# Slicing with a range or vector of indices returns a new `Table`.

@test view(t, 1) == [1, 2, 3]
@test view(t, 1:3) isa Table

# Three helpers give direct access to the underlying flat storage:

@test datarange(t, 1)   == 1:3            # index range into t.data
@test dataview(t, 1)    == [1, 2, 3]      # view into t.data
@test dataview(t, 1:3)  == [1, 2, 3, 2, 3, 5, 8]  # contiguous slice

# `dataiterator` yields `(row, local_position, value)` triples — the workhorse
# for iterating mesh connectivity without per-row allocation.

for (i, j, v) in dataiterator(t)
  @test t[i][j] == v
end

# `generate_data_and_ptrs` reconstructs the flat (data, ptrs) pair from a
# vector of vectors:

data, ptrs = generate_data_and_ptrs(vv)
@test data == [1, 2, 3, 2, 3, 5, 8, 1, 2, 4]
@test ptrs == [1, 4, 6, 8, 8, 11]

# ### Pointer arithmetic helpers
#
# `length_to_ptrs!` converts a lengths vector to an exclusive prefix-sum
# (the standard CSR pointer format). 

lens = [0, 2, 4, 2]
length_to_ptrs!(lens)
@test lens == [1, 3, 7, 9]

# ### inverse_table
#
# `inverse_table(a_to_b)` inverts an index map stored as a `Table`: given a
# cell-to-nodes table, it returns a node-to-cells table.  The output is itself
# a `Table` because multiple cells may share a node.

cell_to_nodes = Table([[1, 2, 3], [1, 2, 3]])   # two cells sharing nodes 1–3
node_to_cells = inverse_table(cell_to_nodes)
@test node_to_cells == [[1, 2], [1, 2], [1, 2]]

# ### append_tables_globally and append_tables_locally
#
# `append_tables_globally` stacks two tables vertically (more rows):

t1 = Table([[1, 2, 3], [2, 3], [5, 8], Int[], [1, 2, 4]])
t2 = Table([[1, 3], [4, 2, 3], Int[], Int[], [1, 2, 4]])
t3 = append_tables_globally(t1, t2)
@test t3 == vcat(t1, t2)

# `append_tables_locally((offsets...), (tables...))` merges corresponding rows
# horizontally, adding an integer offset to each table's entries.  The canonical
# use is building a combined DOF map from per-field contributions: each field's
# local-to-global map is offset by the cumulative DOF count of previous fields.

t4 = append_tables_locally((0, 5), (t1, t2))
@test t4[1] == [1, 2, 3, 6, 8]       # t1[1] ++ (t2[1] .+ 5)
@test t4[2] == [2, 3, 9, 7, 8]       # t1[2] ++ (t2[2] .+ 5)
@test t4[3] == [5, 8]                  # t2[3] is empty, no change
@test t4[5] == [1, 2, 4, 6, 7, 9]

# ### flatten_partition
#
# Given a partition table `a_to_bs` where each `a` owns a set of `b` indices,
# `flatten_partition` inverts it to a flat `b_to_a` vector.

a_to_bs = Table([[1, 2, 3], [7, 8], [4, 5, 6]])
b_to_a  = flatten_partition(a_to_bs)
@test b_to_a == [1, 1, 1, 3, 3, 3, 2, 2]

#
# ## Reindexing arrays lazyly
#
# `Reindex(a)` is a `Map` that performs a single index lookup: `Reindex(a)(i) = a[i]`.
# `lazy_map(Reindex(a), indices)` gives lazy fancy indexing — the result type
# mirrors the structure of `a`.

src = [[1, 2, 4, 5], [2, 4, 6, 7], [4, 3, 5, 1], [2, 3]]
idx = [3, 1, 2]
c   = lazy_map(Reindex(src), idx)
@test c[1] == src[3]
@test c[2] == src[1]

# `Reindex(Fill(...))` → `Fill` (only value is ever read):

c = lazy_map(Reindex(Fill(30.0, 10)), idx)
@test isa(c, Fill)
@test c[1] == 30.0

# `Reindex(CompressedArray(...))` → `CompressedArray` (ptrs reindexed,
# `values` left intact):

ca  = CompressedArray([30, 40, 10, 20, 30], [1, 2, 3, 5, 3, 1, 4, 2])
c   = lazy_map(Reindex(ca), idx)
@test isa(c, CompressedArray)
@test c.values === ca.values   # same values; only ptrs reindexed

# `Reindex(LazyArray(...))` → `LazyArray` (maps and args each reindexed separately,
# preserving the expression tree structure):

la  = lazy_map(-, [1, 2, 3, 5, 3, 1, 4, 2])
c   = lazy_map(Reindex(la), idx)
@test isa(c, LazyArray)
@test c[1] == la[3]

# ### Broadcasting(Reindex(...)) on a Table
#
# A ubiquitous Gridap pattern: apply `Broadcasting(Reindex(global_vals))` to a
# `Table` of local-to-global ids.  For each row (cell), this gathers the global
# values indexed by that row's entries — the cell-local node coordinates,
# DOF values, etc.

p          = VectorValue(1.0, 0.0)
gid_to_val = [p, 2p, 3p, -p, p]
lid_to_gid = Table([[2, 3, 1], [3, 4, 5], [1, 2], Int[]])
ca         = lazy_map(Broadcasting(Reindex(gid_to_val)), lid_to_gid)

@test ca[1] == [2p, 3p,  p]
@test ca[2] == [3p, -p,  p]
@test ca[3] == [ p, 2p]
@test ca[4] == VectorValue{2,Float64}[]

#
# ## PosNegReindex for signed reindexing
#
# `PosNegReindex(values_pos, values_neg)` generalises `Reindex` to a **signed**
# index convention: a positive index `i` reads `values_pos[i]`, and a negative
# index `i` reads `values_neg[-i]`.
#
# This is the mechanism Gridap uses throughout the FE machinery to distinguish
# **free DOFs** (positive global ids) from **Dirichlet-constrained DOFs**
# (negative ids), without any conditional in the inner loop.

values_pos = Float64[40, 30, 10]
values_neg = -Float64[40, 30]
indices    = [1, 3, -1, 2, -2]

c = lazy_map(PosNegReindex(values_pos, values_neg), indices)
@test c[1] ==  40.0   # index  1 → values_pos[1]
@test c[3] == -40.0   # index -1 → values_neg[1]
@test c[5] == -30.0   # index -2 → values_neg[2]

# In the finite element context, `get_cell_dof_ids` returns a `Table` of
# **signed** global DOF ids: positive for free DOFs, negative for Dirichlet.
# Applying `Broadcasting(PosNegReindex(free_dof_vals, dirichlet_vals))` to
# this table extracts the DOF values for every cell — free values from the
# current iterate, Dirichlet values from the boundary data.

# ### PosNegPartition
#
# When the pattern of positive and negative entries is known ahead of time,
# `PosNegPartition` pre-computes the split for more efficient dispatch.
# `PosNegPartition(ipos_to_i, n)` says that among `n` total entries, the
# positions listed in `ipos_to_i` hold positive indices and the rest negative.

part = PosNegPartition([1, 4, 2], 5)   # entries 1,2,4 are positive; 3,5 are negative
c    = lazy_map(PosNegReindex(values_pos, values_neg), part)
r    = [values_pos[i] for i in [1, 3, -1, 2, -2] if i > 0]   # reference positives
@test c[[1, 2, 4]] ≈ r   # the three positive-index slots

# ### lazy_map preserves PosNegReindex structure
#
# When operands are `PosNegReindex`-indexed with the same index pattern,
# `lazy_map` applies the operation **separately** on the positive and negative
# value arrays and returns a new `PosNegReindex`-based structure.  The
# positive and negative branches are never interleaved.

a_pos = Float64[40, 30, 10]; a_neg = -Float64[40, 30]
b_pos = Float64[43, 50, 60]; b_neg = -Float64[41, 30]
a = lazy_map(PosNegReindex(a_pos, a_neg), indices)
b = lazy_map(PosNegReindex(b_pos, b_neg), indices)

c = lazy_map(+, a, b)
@test collect(c) ≈ [a[i] + b[i] for i in 1:length(indices)]

# The decomposition lets Gridap process free-DOF and Dirichlet-DOF contributions
# in separate, homogeneous batches during matrix assembly — avoiding
# branch-per-entry overhead and enabling structure-preserving optimisations.

#
# ## Further reading
#
# - Gridap.jl/src/Arrays/Maps.jl            — Map abstract type, Broadcasting, Operation
# - Gridap.jl/src/Arrays/LazyArrays.jl      — LazyArray, lazy_map, array_cache, getindex!
# - Gridap.jl/src/Arrays/CompressedArrays.jl — CompressedArray and lazy_map specialisations
# - Gridap.jl/src/Arrays/Tables.jl           — Table, inverse_table, append_tables_*, flatten_partition
# - Gridap.jl/src/Arrays/Reindexing.jl       — Reindex, PosNegReindex, PosNegPartition
# - Gridap.jl/src/Arrays/Arrays.jl           — module entry point; full list of exports
