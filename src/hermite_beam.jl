# In this tutorial, we will learn
#  - How to use the Hermite element, the cubic simplex that carries vertex
#    gradients as degrees of freedom
#  - How to solve the Euler--Bernoulli beam, the one setting where that element
#    is $C^1$ and the fourth-order problem is therefore conforming
#  - Why the same element is only $C^0$ in two dimensions, and what it is worth
#    there
#
# ## The element
#
# The **Hermite** element [Ciarlet & Raviart, *Arch. Rational Mech. Anal.* 46
# (1972) 177] is the full cubic space $P_3(K)$ on a simplex
# $K \subset \mathbb{R}^D$, with degrees of freedom
#
# ```math
# \ell^{v,0}(u) = u(v), \qquad
# \ell^{v,i}(u) = \partial_i u(v), \quad i = 1,\dots,D,
# ```
#
# at each vertex $v$, plus the value $u(b)$ at the barycentre $b$ of each
# two-dimensional face. That is $D+1$ degrees of freedom per vertex and one per
# triangular face: 4 on a segment, 10 on a triangle, 20 on a tetrahedron — in
# each case exactly $\dim P_3(K)$.
#
# Because a vertex owns its *whole* gradient, and that gradient is stated in the
# global Cartesian frame, every cell meeting at a vertex shares it. The global
# space is therefore $C^0$ **and** has a single-valued gradient at every vertex.
# It is *not* $C^1$: across an edge the gradients of the two neighbouring cells
# agree at the two endpoints and nowhere else.
#
# Like Morley and Argyris, Hermite is mapped by the plain pullback, and its
# gradient degrees of freedom are not preserved by it — $\nabla u(v) = J^{-T}
# \hat\nabla\hat u(\hat v)$ — so a per-cell change of basis is applied. That is
# internal: `ReferenceFE(hermite, Float64)` is all a user writes. The element
# exists for order 3 only, so the order argument is optional.

using Gridap
using Gridap.ReferenceFEs

# ## The Euler--Bernoulli beam
#
# In one dimension the vertices are the *only* interfaces, so "$C^1$ at the
# vertices" is $C^1$ outright and the element is $H^2$-conforming. This is the
# classical beam element of every structural analysis textbook.
#
# We solve the **clamped beam** under a uniform load: find the deflection $w$ of
# a beam of unit length and unit bending stiffness with
#
# ```math
# w'''' = f \ \text{ in } (0,1), \qquad
# w = w' = 0 \ \text{ at } x = 0, 1 ,
# ```
#
# whose variational form is
#
# ```math
# a(w,v) = \int_0^1 w'' v'' , \qquad \ell(v) = \int_0^1 f\, v .
# ```
#
# For $f \equiv 1$ the deflection is the quartic
#
# ```math
# w(x) = \frac{x^2 (1-x)^2}{24},
# ```
#
# so the midspan deflection is exactly $1/384$.

f = 1.0
w(x) = x[1]^2 * (1 - x[1])^2 / 24

# ## Discrete model and space
#
# A one-dimensional Cartesian model has `SEGMENT` cells, and its face labeling
# already provides a `"boundary"` tag for the two endpoints.
#
# The clamped condition is where this element pays off. A boundary vertex owns
# exactly two degrees of freedom, $w(v)$ and $w'(v)$, and those are precisely
# the two clamped conditions — so `dirichlet_tags="boundary"` imposes the whole
# boundary condition, with nothing left over and nothing missing. The data is
# homogeneous, so `TrialFESpace(V)` suffices.

model = CartesianDiscreteModel((0, 1), (8,))

V = FESpace(model, ReferenceFE(hermite, Float64); dirichlet_tags="boundary")
U = TrialFESpace(V)

reffe = ReferenceFE(SEGMENT, hermite, Float64)
println("dofs per cell: ", num_dofs(reffe))
println("free dofs: ", num_free_dofs(V), "   clamped: ", num_dirichlet_dofs(V))

# ## Weak form and solve
#
# `∇∇(u)` is the Hessian, a $1\times1$ tensor in one dimension, so the bilinear
# form is the double contraction. Unlike the Morley and HHJ tutorials this is
# *not* a broken form: the space really is $H^2$, so the element-wise Hessian
# and the distributional one agree.

Ω = Triangulation(model)
dΩ = Measure(Ω, 8)

a(u, v) = ∫( ∇∇(u) ⊙ ∇∇(v) )dΩ
l(v) = ∫( f * v )dΩ

wh = solve(AffineFEOperator(a, l, U, V))

e = w - wh
println("L2 error: ", sqrt(sum( ∫( e * e )dΩ )))
println("exact midspan deflection: ", 1 / 384)

# ## Convergence
#
# The exact deflection is quartic, so it is not reproduced exactly and we see
# genuine rates: $O(h^4)$ in $L^2$ and $O(h^2)$ in the $H^2$ seminorm, which is
# optimal for a cubic conforming space.

function beam(n)
  model = CartesianDiscreteModel((0, 1), (n,))
  V = FESpace(model, ReferenceFE(hermite, Float64); dirichlet_tags="boundary")
  U = TrialFESpace(V)
  Ω = Triangulation(model)
  dΩ = Measure(Ω, 8)
  a(u, v) = ∫( ∇∇(u) ⊙ ∇∇(v) )dΩ
  l(v) = ∫( f * v )dΩ
  wh = solve(AffineFEOperator(a, l, U, V))
  e = w - wh
  el2 = sqrt(sum( ∫( e * e )dΩ ))
  eh2 = sqrt(sum( ∫( ∇∇(e) ⊙ ∇∇(e) )dΩ ))
  el2, eh2
end

for n in (2, 4, 8, 16)
  el2, eh2 = beam(n)
  println("n = ", n, "   L2 = ", el2, "   H2 = ", eh2)
end

# ## The same element in two dimensions
#
# On triangles the edges are interfaces too, and the element is only $C^0$. The
# fourth-order problem is out of reach — that is what the Morley and Argyris
# tutorials are for — but the vertex gradients are still shared, and that makes
# the global space a proper *subspace* of the continuous cubic Lagrange space.
#
# The two count differently. On a triangulation with $V$ vertices, $E$ edges and
# $C$ cells, cubic Lagrange has one degree of freedom per vertex, two per edge
# and one per cell, while Hermite has three per vertex and one per cell:
#
# ```math
# \dim P_3^{\text{Lagrange}} = V + 2E + C, \qquad
# \dim P_3^{\text{Hermite}} = 3V + C .
# ```
#
# On a large triangulation $E \approx 3V$ and $C \approx 2V$, so Hermite is
# about $5V$ against about $9V$ — a little over half the degrees of freedom for
# the same local polynomial space.

using Gridap.Geometry

model2 = simplexify(CartesianDiscreteModel((0, 1, 0, 1), (8, 8)))
topo = get_grid_topology(model2)
nV, nE, nC = num_faces(topo, 0), num_faces(topo, 1), num_faces(topo, 2)

Vh = FESpace(model2, ReferenceFE(hermite, Float64))
Vl = FESpace(model2, ReferenceFE(lagrangian, Float64, 3))

println("hermite:  ", num_free_dofs(Vh), "   3V + C   = ", 3*nV + nC)
println("lagrange: ", num_free_dofs(Vl), "   V + 2E + C = ", nV + 2*nE + nC)

# ## A cheaper cubic
#
# We solve a reaction--diffusion problem with natural boundary conditions,
#
# ```math
# -\Delta u + u = f \ \text{ in } \Omega = (0,1)^2, \qquad
# \frac{\partial u}{\partial n} = 0 \ \text{ on } \partial\Omega,
# ```
#
# and compare the two spaces on the same meshes. Natural conditions keep the
# comparison clean: a Hermite vertex on the boundary owns its full gradient, so
# `dirichlet_tags` there would pin the *normal* derivative as well as the value,
# which plain Dirichlet data does not determine. (In the beam that was a
# feature — clamping *is* a condition on $w$ and $w'$.)
#
# Both spaces converge at $O(h^4)$ in $L^2$ and $O(h^3)$ in $H^1$ — on these
# coarse meshes the Hermite rates are still climbing towards those values, the
# Lagrange ones are there already. Hermite carries a somewhat larger error
# constant, and about 57% of the degrees of freedom.

uex(x) = cos(pi*x[1]) * cos(pi*x[2]) + cos(2*pi*x[1])
fex(x) = -Δ(uex)(x) + uex(x)

function reaction_diffusion(n, reffe)
  model = simplexify(CartesianDiscreteModel((0, 1, 0, 1), (n, n)))
  V = FESpace(model, reffe)
  Ω = Triangulation(model)
  dΩ = Measure(Ω, 8)
  a(u, v) = ∫( ∇(u) ⋅ ∇(v) + u * v )dΩ
  l(v) = ∫( fex * v )dΩ
  uh = solve(AffineFEOperator(a, l, V, V))
  e = uex - uh
  el2 = sqrt(sum( ∫( e * e )dΩ ))
  eh1 = sqrt(sum( ∫( ∇(e) ⋅ ∇(e) )dΩ ))
  num_free_dofs(V), el2, eh1
end

for n in (2, 4, 8, 16)
  ndh, l2h, h1h = reaction_diffusion(n, ReferenceFE(hermite, Float64))
  ndl, l2l, h1l = reaction_diffusion(n, ReferenceFE(lagrangian, Float64, 3))
  println("n = ", n,
          "   hermite: ", ndh, " dofs, L2 = ", l2h, ", H1 = ", h1h,
          "   lagrange: ", ndl, " dofs, L2 = ", l2l, ", H1 = ", h1l)
end

# ## $C^0$, and $C^1$ at the vertices only
#
# The claim about the space is worth checking rather than asserting. We
# interpolate a smooth function and look at the interpolant from inside each
# cell separately.
#
# Evaluating a `CellField` at a `CellPoint` built from the reference vertices
# gives, for every cell, the gradient at its own three vertices. Gathering those
# per mesh vertex, the *spread* over the cells meeting there is zero for Hermite
# and $O(h^3)$ for Lagrange: the Hermite gradient at a vertex is a single shared
# degree of freedom, the Lagrange one is whatever each cell happens to produce.

using Gridap.CellData
using FillArrays
using LinearAlgebra

g(x) = cos(pi*x[1]) * cos(pi*x[2]) + cos(2*pi*x[1])

Ω2 = Triangulation(model2)
cell_to_vertices = get_faces(topo, 2, 0)
cell_points = CellPoint(
  Fill(get_vertex_coordinates(TRI), num_cells(model2)), Ω2, ReferenceDomain()
)

Λ = SkeletonTriangulation(model2)
dΛ = Measure(Λ, 8)

for (name, space) in (("hermite ", Vh), ("lagrange", Vl))
  gh = interpolate(g, space)
  cell_vertex_grads = ∇(gh)(cell_points)

  at_vertex = [VectorValue{2,Float64}[] for _ in 1:nV]
  for cell in 1:num_cells(model2)
    for (i, vertex) in enumerate(cell_to_vertices[cell])
      push!(at_vertex[vertex], cell_vertex_grads[cell][i])
    end
  end
  spread = maximum(at_vertex) do vals
    maximum(v -> norm(v - first(vals)), vals)
  end

  jump_u = sqrt(sum( ∫( jump(gh) * jump(gh) )dΛ ))
  jump_g = sqrt(sum( ∫( jump(∇(gh)) ⋅ jump(∇(gh)) )dΛ ))
  println(name, "  vertex gradient spread: ", spread,
          "   [[u]]: ", jump_u, "   [[grad u]]: ", jump_g)
end

# The skeleton jumps say the rest: the value jump is at round-off for both
# spaces — they are $C^0$ — while the gradient jump is $O(1)$ for both. Hermite
# is not $C^1$, and its gradient jump is in fact the larger of the two here. The
# vertex gradients are shared, not the edge ones.
#
# For the Hermite interpolant that shared value is also the *exact* one, since
# the degree of freedom is $\partial_i u(v)$ itself:

gh = interpolate(g, Vh)
cell_vertex_grads = ∇(gh)(cell_points)
vertex_coords = get_vertex_coordinates(topo)

err = maximum(
  norm(cell_vertex_grads[cell][i] - ∇(g)(vertex_coords[vertex]))
  for cell in 1:num_cells(model2)
  for (i, vertex) in enumerate(cell_to_vertices[cell])
)
println("largest vertex gradient error of the interpolant: ", err)

# ## Visualisation

mkpath("output_path")
writevtk(Ω2, "output_path/hermite", cellfields=["u" => gh, "grad" => ∇(gh)])
