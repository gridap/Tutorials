# In this tutorial, we will learn
#
#  - How to define a `ReferenceFEName` type to tag a new element family
#  - How to construct a polynomial prebasis using `MonomialBasis`
#  - How to encode moment-based DOFs as weighted integrals over element faces
#  - How to implement a moment integrand σ(φ,μ,ds) using Gridap's `Field` algebra
#  - How to assemble a complete `ReferenceFE` via `MomentBasedReferenceFE`
#  - How to plug the element into the standard `ReferenceFE` dispatch system
#  - How to verify the implementation against the built-in CR element and via convergence

using Gridap
using Gridap.ReferenceFEs
using Gridap.Polynomials
using Gridap.Fields

#
# ## What constitutes a ReferenceFE
#
# A `ReferenceFE{D}` in Gridap encodes everything the assembly loop needs to
# know about a single element type on a reference polytope:
#
# | Field | Role |
# |-------|------|
# | `prebasis` | polynomial space $\mathcal{P}$, $\dim\mathcal{P} = n$ |
# | `dofs` | $n$ linear functionals $\sigma_1,\ldots,\sigma_n$ on $\mathcal{P}$ |
# | `shapefuns` | dual basis $\varphi_j$: $\sigma_i(\varphi_j)=\delta_{ij}$ |
# | `face_own_dofs` | which DOF indices belong to each face of the polytope |
# | `conformity` | how adjacent cells share DOFs across facets |
#
# The shape functions are the columns of $\mathbf{M}^{-1}$ where
# $M_{ij}=\sigma_i(p_j)$ is the Vandermonde matrix of the prebasis $\{p_j\}$.
# Gridap's `MomentBasedReferenceFE` constructor handles the inversion and
# assembles the full struct, provided you supply the prebasis and a description
# of the DOFs as moment integrals over faces.

#
# ## The Crouzeix–Raviart element
#
# The Crouzeix–Raviart (CR) element [1] is the canonical lowest-order
# nonconforming scalar element on simplices. On a $D$-simplex $K$ it pairs
# $\mathcal{P}_1(K)$ (dimension $D+1$) with one DOF per facet $F_i$:
#
# ```math
# \sigma_{F_i}(\varphi) = \frac{1}{|F_i|}\int_{F_i}\varphi\;dF,
# \qquad i=1,\ldots,D+1.
# ```
#
# A simplex has exactly $D+1$ facets, so the set is unisolvent.
# Global CR functions are only weakly continuous across inter-element boundaries
# — adjacent cells agree on facet averages, not on pointwise values.
# This places global functions in the broken space $H^1_h(\mathcal{T}_h)$
# rather than $H^1(\Omega)$.

#
# ## Step 1 — The ReferenceFEName type
#
# Every element family is identified by a singleton subtype of `ReferenceFEName`.
# The type carries no data; it is a pure dispatch tag.
# The constants you already know from high-level drivers — `lagrangian`,
# `raviart_thomas`, `nedelec` — are exactly this: singleton instances of their
# respective `ReferenceFEName` subtypes, used as the second argument to
# `ReferenceFE(polytope, name, T, order)`.

struct MyCrouzeixRaviart <: ReferenceFEName end
const my_crouzeix_raviart = MyCrouzeixRaviart()

#
# ## Step 2 — The polynomial prebasis
#
# `MonomialBasis(Val(D), T, order, filter)` builds the basis of scalar monomials
# in `D` variables of type `T` up to degree `order`, keeping only monomials
# $x^\alpha$ for which `filter(α,order)` is true.  The standard
# complete-polynomial filter `Polynomials._p_filter` accepts $|\alpha|\le\text{order}$,
# spanning $\mathcal{P}_\text{order}$.  Two other built-in filters are
# `Polynomials._q_filter` (tensor-product $\mathcal{Q}_\text{order}$: each
# component $\alpha_i\le\text{order}$, used for Lagrangian elements on quads/hexes)
# and `Polynomials._s_filter` (serendipity space: drops high-degree corner monomials
# to reduce the DOF count on quads/hexes while retaining full approximation order).

D = 2
T = Float64
p = TRI

prebasis = MonomialBasis(Val(D), T, 1, Polynomials._p_filter)

# On a triangle $\mathcal{P}_1$ has dimension $D+1=3$, spanned by
# $\{1,\,x_1,\,x_2\}$:

length(prebasis)

# Evaluating at the centroid $(1/3,1/3)$ of the reference triangle confirms
# the three monomials:

evaluate(prebasis, [Point(1/3, 1/3)])

#
# ## Step 3 — The moment test basis on facets
#
# Each DOF integral $\int_F \varphi\cdot\mu\,dF$ involves a test polynomial
# $\mu$ defined on the facet $F$.  For CR, $\mu=1$ ($\mathcal{P}_0$), so we
# need one constant basis function per facet.  Since a facet is a
# $(D-1)$-simplex, we build a `MonomialBasis` in $D-1$ variables of degree 0:

fb = MonomialBasis(Val(D-1), T, 0, Polynomials._p_filter)
length(fb)

# There is exactly one moment per facet, so the total DOF count is 3 — matching
# the three edges of `TRI` and the dimension of $\mathcal{P}_1$.

#
# ## Step 4 — The moment integrand
#
# A moment descriptor is a triplet `(face_range, σ, μ)` where
# - `face_range` selects which faces of the polytope carry the moments
#   (an index range into the global face list of `p`),
# - `σ(φ,μ,ds)` is the integrand (linear in both `φ` and `μ`),
# - `μ` is the test basis on the face.
#
# When `MomentBasedReferenceFE` builds the DOF basis it calls `σ` with:
# - `φ`: the prebasis restricted and pulled back to the current face,
# - `μ`: the test basis in the face reference domain,
# - `ds`: a `FaceMeasure` carrying `cpoly` (cell polytope), `face` (local face
#   index), and the quadrature and face-to-cell coordinate map.
#
# `σ` must return a `Field` (or array of `Field`s) representing the integrand.
# No numerical values are computed here; the framework evaluates the result at
# quadrature points internally.
#
# The CR integrand $\sigma_F(\varphi) = |F|^{-1}\int_F\varphi\,dF$ is:

function fmom(φ, μ, ds)
  D = num_dims(ds.cpoly)
  facet_meas = ReferenceFEs._get_dfaces_measure(ds.cpoly, D-1)    # vector of facet measures |F|
  inv_meas   = Fields.ConstantField(1 / facet_meas[ds.face])
  φμ = Broadcasting(Operation(⋅))(φ, μ)              # pointwise product of Field arrays
  Broadcasting(Operation(*))(φμ, inv_meas)            # scale by 1/|F|
end

# `_get_dfaces_measure` is an internal Gridap helper that returns the
# $d$-volume of each $d$-face of a polytope (edge lengths, face areas, etc.).
#
# `Broadcasting(Operation(f))` lifts a binary scalar operation `f` to act
# element-wise on arrays of `Field`s, returning a new array of `Field`s.
# The computation is deferred: the resulting expression tree is only evaluated
# at quadrature points when `MomentBasedDofBasis` assembles the Vandermonde matrix.

#
# ## Step 5 — Assembling the ReferenceFE
#
# `get_dimrange(p, d)` returns the range of face indices for all $d$-faces of
# `p`; passing `D-1` selects all edges (facets of a triangle).

moments = Tuple[
  (get_dimrange(p, D-1), fmom, fb),
]

my_cr = ReferenceFEs.MomentBasedReferenceFE(my_crouzeix_raviart, p, prebasis, moments, L2Conformity())

# Inspecting the face-DOF ownership confirms one DOF per edge (faces 4–6 in
# the TRI face numbering, after 3 vertices and the cell interior) and none on
# vertices or interior:

get_face_own_dofs(my_cr)

# The CR shape functions are the unique degree-1 polynomials whose average
# over each edge is a Kronecker delta: $\sigma_i(\varphi_j)=\delta_{ij}$.
# We verify this by evaluating the DOF functionals against the shape functions
# — the result should be the $3\times 3$ identity:

dof_basis = get_dof_basis(my_cr)
shp       = get_shapefuns(my_cr)
evaluate(dof_basis, shp)

# It can also be instructive to compare the shape functions against the prebasis.
# The Vandermonde matrix $M_{ij} = \sigma_i(p_j)$ (DOFs applied to monomials)
# encodes how the shape functions are formed as linear combinations of monomials:

evaluate(dof_basis, prebasis)

#
# ## Comparing with the built-in implementation
#
# Gridap ships its own `CrouzeixRaviartRefFE`.  Both DOF bases must produce the
# same Vandermonde matrix when applied to the common prebasis:

cr_builtin   = CrouzeixRaviartRefFE(Float64, TRI, 1)
dofs_builtin = get_dof_basis(cr_builtin)

M_mine   = evaluate(dof_basis,   prebasis)
M_ref    = evaluate(dofs_builtin, prebasis)
@assert M_mine ≈ M_ref

#
# ## Step 6 — Registering the dispatch hook
#
# For the element to be usable through the standard `ReferenceFE(polytope, name, T, order)`
# API (and hence in `FESpace`, `TestFESpace`, etc.) we add one method.

function ReferenceFE(p::Polytope, ::MyCrouzeixRaviart, ::Type{T}, order) where T
  @assert is_simplex(p) && order == 1 "MyCrouzeixRaviart is only defined for simplices at order 1"
  D   = num_dims(p)
  pre = MonomialBasis(Val(D), T, order, Polynomials._p_filter)
  fb  = MonomialBasis(Val(D-1), T, 0, Polynomials._p_filter)
  function σ(φ, μ, ds)
    d     = num_dims(ds.cpoly)
    fm    = ReferenceFEs._get_dfaces_measure(ds.cpoly, d-1)
    scale = Fields.ConstantField(1 / fm[ds.face])
    Broadcasting(Operation(*))(Broadcasting(Operation(⋅))(φ, μ), scale)
  end
  moms = Tuple[(get_dimrange(p, D-1), σ, fb)]
  ReferenceFEs.MomentBasedReferenceFE(my_crouzeix_raviart, p, pre, moms, L2Conformity())
end

# We also need a `get_face_own_dofs` override for `L2Conformity`.  Without it,
# the default `L2Conformity` dispatch places all DOFs on the cell interior,
# which would make adjacent cells independent and break the nonconforming
# inter-element coupling that CR relies on.

function ReferenceFEs.get_face_own_dofs(reffe::GenericRefFE{MyCrouzeixRaviart}, ::L2Conformity)
  get_face_own_dofs(reffe)
end

# This one-liner redirects to the stored `face_own_dofs` (the moment-derived
# facet ownership), so that when the `FESpace` machinery builds the global DOF
# numbering, it recognises that the edge DOF is shared between the two adjacent
# cells — exactly the nonconforming H1 interpretation.

#
# ## Verification: Poisson convergence
#
# We solve
# ```math
# -\Delta u = f \quad\text{in } \Omega=[0,1]^2, \qquad u=0 \quad\text{on }\partial\Omega,
# ```
# with manufactured solution $u(x)=\sin(\pi x_1)\sin(\pi x_2)$ and
# $f=2\pi^2 u$.  For the nonconforming CR discretisation (broken $H^1$ bilinear
# form, no penalty), the expected convergence rates are $O(h^2)$ in $L^2$ and
# $O(h)$ in the broken $H^1$ seminorm [1].

u_ex(x) = sin(π*x[1]) * sin(π*x[2])
f_ex(x) = 2π^2 * sin(π*x[1]) * sin(π*x[2])

function solve_cr_poisson(n)
  model = simplexify(CartesianDiscreteModel((0,1,0,1),(n,n)))
  reffe = ReferenceFE(TRI, my_crouzeix_raviart, Float64, 1)
  V = TestFESpace(model, reffe; conformity=:L2, dirichlet_tags="boundary")
  U = TrialFESpace(V, u_ex)

  Ω  = Triangulation(model)
  dΩ = Measure(Ω, 4)

  a(u,v) = ∫( ∇(u)⊙∇(v) )dΩ
  l(v)   = ∫( f_ex * v )dΩ

  op = AffineFEOperator(a, l, U, V)
  uh = solve(op)

  e  = u_ex - uh
  el2 = sqrt(sum( ∫(e*e)dΩ ))
  eh1 = sqrt(sum( ∫(∇(e)⊙∇(e))dΩ ))
  return el2, eh1
end

# Run on a sequence of successively refined meshes and compute convergence rates:

ns   = [4, 8, 16]
errs = [solve_cr_poisson(n) for n in ns]

for i in 2:length(ns)
  r_l2 = log(errs[i-1][1]/errs[i][1]) / log(2)
  r_h1 = log(errs[i-1][2]/errs[i][2]) / log(2)
  println("n=$(ns[i]):  L² rate ≈ $(round(r_l2,digits=2)),  broken H¹ rate ≈ $(round(r_h1,digits=2))")
end

#
# ## References
#
# [1] M. Crouzeix, P.-A. Raviart. *Conforming and nonconforming finite element
#     methods for solving the stationary Stokes equations I.*
#     RAIRO Anal. Numér., 7(R-3):33–75, 1973.
#     doi:[10.1051/m2an/197307R300331](http://dx.doi.org/10.1051/m2an/197307R300331)

#
# ## Further reading
#
# The following source files in `Gridap.jl` are the most relevant to browse
# alongside this tutorial:
#
# - `src/ReferenceFEs/MomentBasedReferenceFEs.jl` — `MomentBasedDofBasis`,
#   `FaceMeasure`, and `MomentBasedReferenceFE`; the core machinery used here.
# - `src/ReferenceFEs/CrouzeixRaviartRefFEs.jl` — the built-in CR element
#   whose logic this tutorial re-derives step by step.
# - `src/ReferenceFEs/RaviartThomasRefFEs.jl` — a more involved moment-based
#   element: vector-valued prebasis, $H(\text{div})$ conformity, and normal-trace
#   moments instead of plain averages.
# - `src/ReferenceFEs/NedelecRefFEs.jl` — $H(\text{curl})$ analogue of RT;
#   tangential-trace moments, useful as a contrast to the normal-trace case.
# - `src/ReferenceFEs/LagrangianRefFEs.jl` — point-evaluation DOFs (the
#   `LagrangianDofBasis` counterpart to `MomentBasedDofBasis`).
# - `src/ReferenceFEs/ReferenceFEInterfaces.jl` — abstract interface:
#   `ReferenceFE`, `ReferenceFEName`, `Conformity`, and `GenericRefFE`.
# - `src/Polynomials/MonomialBases.jl` — `MonomialBasis` and the three
#   polynomial filters (`_p_filter`, `_q_filter`, `_s_filter`).
