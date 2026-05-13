# In this tutorial, we will learn
#
#  - What the type parameters of `MultiValue{S,T,N,L}` encode and why `L` may be
#    strictly less than `prod(S)`
#  - How to define a concrete `MultiValue` subtype with constrained storage
#  - How to implement the `getindex` and array-conversion interface so that generic
#    operations work without further code
#  - Which arithmetic, contraction, and linear-algebra operations Gridap provides
#    for free once the interface is satisfied
#  - How to add type-preserving operations that exploit a type's algebraic structure
#  - What the built-in `MultiValue` subtypes are and how their storage layouts differ

using Gridap
using Gridap.TensorValues
using LinearAlgebra: I
using Base: @propagate_inbounds
using Test

#
# ## The `MultiValue` type hierarchy
#
# `Gridap.TensorValues` revolves around the abstract type
#
# ```julia
# MultiValue{S, T, N, L} <: Number
# ```
#
# Its four type parameters mirror those of `StaticArrays.SArray`:
#
# | Parameter | Meaning | Example |
# |-----------|---------|---------|
# | `S` | `Tuple` type encoding the full tensor shape | `Tuple{3,3}` for a 3×3 matrix |
# | `T` | scalar component type | `Float64` |
# | `N` | tensor order (length of `S`) | `2` |
# | `L` | number of **stored** components | ≤ `prod(S)` |
#
# The crucial design freedom is `L ≤ prod(S)`.  A `TensorValue{3,3}` stores all 9
# entries; a `SymTensorValue{3}` stores only the 6 upper-triangle entries and
# reconstructs the lower triangle via symmetry on `getindex`; a
# `SkewSymTensorValue{3}` stores 3 entries, recovers the diagonal (zero) and
# lower triangle (negation) at index time.
#
# Subtypes of `MultiValue` subtype `Number`, not `AbstractArray`.  This makes
# them behave as scalars under Julia's broadcasting: an array of `VectorValue`s
# participates in the same broadcast machinery as an array of `Float64`s.
#
# ## Implementing a custom `CirculantValue` tensor type
#
# A **circulant matrix** is fully determined by its first row
# $c = (c_1, c_2, \ldots, c_D)$.  Every subsequent row is a cyclic shift:
#
# ```math
# A_{ij} = c_{\;\mathrm{mod}_1(j - i + 1,\; D)}
# ```
#
# For $D = 3$:
#
# ```math
# A = \begin{pmatrix} c_1 & c_2 & c_3 \\ c_3 & c_1 & c_2 \\ c_2 & c_3 & c_1 \end{pmatrix}
# ```
#
# A $D \times D$ circulant is a `Tuple{D,D}`-shaped second-order tensor whose
# $D^2$ entries depend on only $D$ independent values, so `L = D`.  Circulant
# matrices arise in periodic-boundary convolution operators and are the matrices
# simultaneously diagonalised by the discrete Fourier transform.

struct CirculantValue{D,T} <: MultiValue{Tuple{D,D},T,2,D}
  data::NTuple{D,T}
  function CirculantValue{D,T}(data::NTuple{D,T}) where {D,T}
    new{D,T}(data)
  end
end

# The `.data` field holds the first row — the $D$ independent components.
# Gridap's generic machinery **assumes this field exists** and treats it as the
# canonical independent-component vector, so `get_indep_components` and `Tuple`
# already work correctly without any override.
#
# ### Constructors
#
# We follow the same pattern as `VectorValue` and `SymTensorValue`: a primary
# `NTuple` form, a promoted `Tuple` form, and a vararg convenience form.

CirculantValue(data::NTuple{D,T}) where {D,T}            = CirculantValue{D,T}(data)
CirculantValue{D}(data::NTuple{D,T}) where {D,T}         = CirculantValue{D,T}(data)
CirculantValue{D,T1}(data::NTuple{D,T2}) where {D,T1,T2} = CirculantValue{D,T1}(NTuple{D,T1}(data))

CirculantValue(data::Tuple)                      = CirculantValue(promote(data...))
CirculantValue{D}(data::Tuple) where {D}         = CirculantValue{D}(promote(data...))
CirculantValue{D,T}(data::Tuple) where {D,T}     = CirculantValue{D,T}(NTuple{D,T}(data))

CirculantValue(data::Number...)                   = CirculantValue(data)
CirculantValue{D}(data::Number...) where {D}      = CirculantValue{D}(data)
CirculantValue{D,T}(data::Number...) where {D,T}  = CirculantValue{D,T}(data)

# ### Completing the mandatory interface
#
# Three methods form the non-negotiable minimum; without any one of them,
# core generic operations silently break.

import Gridap.TensorValues: change_eltype, num_indep_components

# **`change_eltype`** — returns the parameterised type with a different scalar
# component type.  The generic fallback `change_eltype(::Type{<:Number>, T) = T`
# would return `Float64` instead of `CirculantValue{D,Float64}`, making scalar
# multiplication (`2.0 * a`) crash when it tries `Float64((2.0, 4.0, 6.0))`.

change_eltype(::Type{<:CirculantValue{D}}, ::Type{T2}) where {D,T2} = CirculantValue{D,T2}

# **`num_indep_components`** — the fallback `num_components(V) = prod(S) = D²`
# gives the wrong count, breaking `zero(a)` (which calls
# `V(tfill(zero(T), Val(num_indep_components(V))))` and tries to build a
# `CirculantValue` with D² zeros) and `conj`/`real`/`imag` (which slice
# `Tuple(a)[1:Li]` with `Li = D²` even though `.data` has only D entries).

num_indep_components(::Type{<:CirculantValue{D}}) where {D} = D

# **`get_array`** — many higher-level operations (contraction, determinant,
# eigenvalues, `outer`) delegate to `get_array`, which must return an `SMatrix`.
# We expand all $D^2$ entries from the $D$-element first row.  The `ntuple`
# expression is type-stable and produces zero allocations:

import Gridap.TensorValues: get_array

function get_array(arg::CirculantValue{D,T}) where {D,T}
  # Expand the D independent entries into D² column-major data, then delegate
  # to TensorValue's own get_array (which returns the StaticArrays SMatrix).
  data = ntuple(Val(D*D)) do idx
    j = (idx - 1) ÷ D + 1   # 1-based column index
    i = (idx - 1) % D + 1   # 1-based row index
    @inbounds arg.data[mod1(j - i + 1, D)]
  end
  get_array(TensorValue{D,D,T}(data))
end

# ### `getindex`
#
# For scalar two-index access, the circulant formula maps any `(i,j)` to a
# single lookup in the $D$-element `.data` tuple:

@propagate_inbounds function Base.getindex(a::CirculantValue{D}, i::Integer, j::Integer) where {D}
  @boundscheck checkbounds(a, i, j)
  @inbounds a.data[mod1(j - i + 1, D)]
end

# Confirm the layout for a concrete 3×3 example:

a = CirculantValue(1.0, 2.0, 3.0)

@test a[1,1] == 1.0  &&  a[1,2] == 2.0  &&  a[1,3] == 3.0
@test a[2,1] == 3.0  &&  a[2,2] == 1.0  &&  a[2,3] == 2.0
@test a[3,1] == 2.0  &&  a[3,2] == 3.0  &&  a[3,3] == 1.0

@test Matrix(get_array(a)) == [1.0 2.0 3.0
                                3.0 1.0 2.0
                                2.0 3.0 1.0]

# ## Operations provided for free
#
# Because `CirculantValue` satisfies the `MultiValue` interface, Gridap's generic
# implementations for arithmetic, contractions, and linear algebra apply
# immediately — no additional code required.
#
# ### Arithmetic on independent components
#
# Addition, subtraction, and scalar multiplication act on the $D$ stored entries
# and return a new `CirculantValue`.  The cost is $O(D)$, not $O(D^2)$.

b = CirculantValue(4.0, 5.0, 6.0)

@test a + b === CirculantValue(5.0, 7.0, 9.0)
@test b - a === CirculantValue(3.0, 3.0, 3.0)
@test 2.0 * a === CirculantValue(2.0, 4.0, 6.0)
@test a / 2.0 === CirculantValue(0.5, 1.0, 1.5)

# `zero` constructs the zero circulant (all-zero first row) via
# `num_indep_components`, and works without any override:

@test iszero(zero(a))
@test zero(CirculantValue{3,Float64}) === CirculantValue(0.0, 0.0, 0.0)

# ### Frobenius inner product and norm
#
# `inner(a, b)` or `a ⊙ b` is a full contraction over all index pairs,
# accessing all $D^2$ entries through `getindex`:

@test a ⊙ a ≈ sum(a[i,j]^2 for i in 1:3, j in 1:3)
@test norm(a) ≈ sqrt(a ⊙ a)

# ### Outer product
#
# `a ⊗ b` yields a fourth-order `HighOrderTensorValue{Tuple{3,3,3,3}}`:

c = a ⊗ b
@test c isa HighOrderTensorValue
@test c[1,2,1,3] ≈ a[1,2] * b[1,3]

# ### Matrix–vector contraction
#
# Contracting `a ⋅ v` against a `VectorValue` performs the matrix–vector product
# and returns a `VectorValue`:

v = VectorValue(1.0, 0.0, 0.0)
w = a ⋅ v
@test w isa VectorValue{3,Float64}
@test w ≈ VectorValue(a[1,1], a[2,1], a[3,1])   # multiplication by the first unit vector extracts the first column

# ### Trace and determinant
#
# Both delegate to `getindex` or `get_array` internally.  For a circulant,
# every diagonal entry equals $c_1$, so $\mathrm{tr}(A) = D \cdot c_1$:

@test tr(a) ≈ 3 * a.data[1]
@test det(a) ≈ det(Matrix(get_array(a)))

# ## Type-preserving extensions
#
# Generic contractions return the most general matching type.  Contracting two
# `CirculantValue`s with `⋅` uses `contracted_product`, which returns a
# `TensorValue` — the circulant structure is lost:

ab = a ⋅ b
@test ab isa TensorValue{3,3,Float64}

# The matrix entries are correct, but all $D^2 = 9$ values are now stored
# instead of $D = 3$.  We recover the compact form by exploiting the core
# algebraic property: **the product of two circulants is again circulant**, with
# first row equal to the cyclic convolution of the two input rows:
#
# ```math
# (A B)_{1,n} = \sum_{k=1}^{D} c^a_k \; c^b_{\mathrm{mod}_1(n-k+1,\,D)}
# ```

function circ_mul(a::CirculantValue{D,Ta}, b::CirculantValue{D,Tb}) where {D,Ta,Tb}
  T = promote_type(Ta, Tb)
  row = ntuple(Val(D)) do n
    sum(a.data[k] * b.data[mod1(n - k + 1, D)] for k in 1:D)
  end
  CirculantValue{D,T}(row)
end

@test circ_mul(a, b) isa CirculantValue{3,Float64}
@test get_array(circ_mul(a, b)) ≈ get_array(a) * get_array(b)

# `circ_mul` stores only $D$ values and computes in $O(D^2)$ — the same
# arithmetic as the general product, but with $D$ rather than $D^2$ storage.
#
# ### Transpose
#
# For a circulant with first row $(c_1, c_2, \ldots, c_D)$, the transpose has
# first row $(c_1, c_D, c_{D-1}, \ldots, c_2)$ — position 1 is fixed and
# positions 2 through $D$ are reversed.  Gridap's fallback for
# `MultiValue{Tuple{D,D}}` is `@notimplemented`, so we provide a specialisation:

function Base.transpose(a::CirculantValue{D,T}) where {D,T}
  data = ntuple(Val(D)) do k
    k == 1 ? a.data[1] : a.data[D - k + 2]
  end
  CirculantValue{D,T}(data)
end

at = transpose(a)
@test at isa CirculantValue{3,Float64}
@test get_array(at) ≈ transpose(Matrix(get_array(a)))

# A circulant is symmetric iff its first row is a palindrome:

s = CirculantValue(1.0, 2.0, 2.0)
@test transpose(s) == s

# ### Mixed-precision arithmetic (optional extension)
#
# By default, `promote_rule` for two different-T `MultiValue`s returns `Union{}`
# — meaning Julia raises a method error rather than trying to promote.  Adding
# `promote_rule` and a matching `convert` unlocks operations like
# `CirculantValue{3,Float64} + CirculantValue{3,Int64}`.
#
# The `convert` must be explicit: the generic `convert(::Type{T<:Number}, x::Number) = T(x)`
# would call `CirculantValue{3,Float64}(x::CirculantValue{3,Int64})`, which
# matches the vararg constructor with a 1-element argument and then fails
# constructing `NTuple{3,Float64}` from a 1-element tuple.

Base.promote_rule(::Type{<:CirculantValue{D,Ta}}, ::Type{<:CirculantValue{D,Tb}}) where {D,Ta,Tb} =
  CirculantValue{D, promote_type(Ta,Tb)}

Base.convert(::Type{<:CirculantValue{D,T}}, arg::CirculantValue{D}) where {D,T} =
  CirculantValue{D,T}(Tuple(arg))
Base.convert(::Type{<:CirculantValue{D,T}}, arg::CirculantValue{D,T}) where {D,T} = arg

@test a + CirculantValue(1, 0, 0) === CirculantValue(2.0, 2.0, 3.0)

# ### Identity
#
# The $D \times D$ identity matrix is circulant with first row $(1, 0, \ldots, 0)$.
# Overriding `one` lets generic code that calls `one(a)` or `one(CirculantValue{D,T})`
# get the compact representation rather than an error:

function Base.one(::Type{V}) where {D, T, V <: CirculantValue{D,T}}
  CirculantValue{D,T}(ntuple(i -> i == 1 ? one(T) : zero(T), Val(D)))
end
Base.one(a::CirculantValue) = one(typeof(a))

id = one(CirculantValue{3,Float64})
@test get_array(id) ≈ Matrix(1.0*I, 3, 3)
@test circ_mul(a, id) == a

# ## Built-in `MultiValue` types
#
# Gridap ships nine concrete subtypes covering the tensors most common in
# continuum mechanics and electromagnetics.
#
# | Type | Shape `S` | `L` | Storage |
# |------|-----------|-----|---------|
# | `VectorValue{D,T}` | `Tuple{D}` | `D` | all D components |
# | `TensorValue{D1,D2,T}` | `Tuple{D1,D2}` | `D1·D2` | all, column-major |
# | `SymTensorValue{D,T}` | `Tuple{D,D}` | `D(D+1)/2` | upper triangle, row ≤ col |
# | `SymTracelessTensorValue{D,T}` | `Tuple{D,D}` | `D(D+1)/2 − 1` | upper triangle; last diagonal recovered as `−(sum of others)` |
# | `SkewSymTensorValue{D,T}` | `Tuple{D,D}` | `D(D−1)/2` | strict upper triangle; diagonal zero, lower triangle negated |
# | `ThirdOrderTensorValue{D1,D2,D3,T}` | `Tuple{D1,D2,D3}` | `D1·D2·D3` | all, column-major |
# | `HighOrderTensorValue{S,T,N}` | any `S` | `prod(S)` | all, `SArray` |
# | `SymFourthOrderTensorValue{D,T}` | `Tuple{D,D,D,D}` | `(D(D+1)/2)²` | both index pairs symmetric: `ijkl=jikl=ijlk` |
#
# The independent-component counts for $D = 3$:

@test num_indep_components(VectorValue{3,Float64})             == 3
@test num_indep_components(TensorValue{3,3,Float64})           == 9
@test num_indep_components(SymTensorValue{3,Float64})          == 6
@test num_indep_components(SymTracelessTensorValue{3,Float64}) == 5
@test num_indep_components(SkewSymTensorValue{3,Float64})      == 3
@test num_indep_components(ThirdOrderTensorValue{2,2,2,Float64}) == 8
@test num_indep_components(SymFourthOrderTensorValue{2,Float64}) == 9

# **`VectorValue{D,T}`** — a plain $D$-vector.

u = VectorValue(1.0, 2.0, 3.0)
@test u ⋅ u ≈ 14.0

# **`TensorValue{D1,D2,T}`** — a general $D_1 \times D_2$ matrix with no symmetry
# imposed.  The constructor fills entries in column-major order.

F = TensorValue(1.0, 0.0, 0.0,   # column 1
                0.0, 2.0, 0.0,   # column 2
                0.0, 0.0, 3.0)   # column 3
@test F[1,1] == 1.0  &&  F[2,2] == 2.0  &&  F[3,3] == 3.0
@test det(F) ≈ 6.0

# **`SymTensorValue{D,T}`** — symmetric second-order tensor; stores only the
# upper triangle.  Entry (i,j) with i > j is recovered as (j,i).

σ = SymTensorValue(1.0, 0.5, 0.0,   # row 1: σ₁₁, σ₁₂, σ₁₃
                        2.0, 0.0,   #        σ₂₂, σ₂₃
                             3.0)   #        σ₃₃
@test σ[2,1] == σ[1,2] == 0.5

# **`SkewSymTensorValue{D,T}`** — anti-symmetric; diagonal always zero, lower
# triangle recovered as the negation of the upper triangle.

Ω = SkewSymTensorValue(0.1, 0.2, 0.3)   # upper-triangle entries for D=3
@test Ω[2,1] == -Ω[1,2]
@test iszero(Ω[1,1])

# **`SymTracelessTensorValue{D,T}`** (alias `QTensorValue`) — symmetric and
# trace-free; used for liquid-crystal order-parameter tensors.  The final
# diagonal entry is not stored; it is recovered as minus the sum of the others.

Q = SymTracelessTensorValue(0.3, 0.0, -0.1, 0.0, 0.0)   # 5 entries for D=3
@test tr(Q) ≈ 0.0

# **`ThirdOrderTensorValue{D1,D2,D3,T}`** — third-order tensor, alias for
# `HighOrderTensorValue{Tuple{D1,D2,D3}}`, no symmetry assumed.

G = ThirdOrderTensorValue{2,2,2,Float64}(ntuple(i -> Float64(i), 8))
@test size(G) == (2, 2, 2)

# **`SymFourthOrderTensorValue{D,T}`** — fourth-order tensor with the two minor
# symmetries $\mathbb{C}_{ijkl} = \mathbb{C}_{jikl} = \mathbb{C}_{ijlk}$ that
# characterise the elasticity tensor.

𝕔 = SymFourthOrderTensorValue{2,Float64}(ntuple(i -> Float64(i), 9))
@test 𝕔[1,2,1,1] == 𝕔[2,1,1,1]   # minor symmetry ij↔ji

# Contracting the elasticity tensor with a symmetric strain gives a stress:

ε = SymTensorValue(0.1, 0.0, 0.2)   # 2×2 symmetric strain
τ = 𝕔 ⋅² ε                          # double contraction: stress
@test τ isa SymTensorValue{2,Float64}
