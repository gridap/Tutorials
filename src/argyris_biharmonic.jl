# In this tutorial, we will learn
#  - How to use the Argyris element, the classical $C^1$-conforming triangle
#  - How to impose clamped boundary conditions when the degrees of freedom are derivatives
#  - How to measure the $C^1$ continuity of a discrete space
#
# ## Problem statement
#
# We solve the biharmonic problem
#
# ```math
# \Delta^2 u = f \ \text{ in } \Omega = (0,1)^2,
# ```
#
# with a manufactured solution, using an $H^2$-*conforming* discretisation. The
# variational form is
#
# ```math
# a(u,v) = \int_\Omega D^2 u : D^2 v, \qquad \ell(v) = \int_\Omega f\, v,
# ```
#
# and, unlike in the Morley tutorial, here $D^2 u$ really is the distributional
# Hessian: the discrete space is a subspace of $H^2(\Omega)$.
#
# ## The element
#
# The **Argyris** triangle is the quintic space $P_5(K)$, of dimension 21, with
# per vertex the value, the two first derivatives and the three second
# derivatives, and per edge one normal-derivative moment:
#
# ```math
# u(v),\quad \partial_i u(v),\quad \partial_{ij} u(v), \qquad \int_e \nabla u\cdot n .
# ```
#
# Sharing all six vertex quantities between the elements meeting at a vertex, and
# the normal-derivative moment between the two elements sharing an edge, makes the
# assembled space $C^1$. It is the cheapest $C^1$ triangle there is.
#
# Two consequences worth noting. The vertex degrees of freedom are *point
# evaluations of derivatives*, so they are not preserved by the pullback and the
# element needs a cell-dependent change of basis — Gridap does this internally.
# And because those degrees of freedom include second derivatives, a boundary
# condition on `dirichlet_tags` constrains the boundary Hessian too, which is why
# below we interpolate the exact solution rather than setting zero.

using Gridap

# ## Discrete model
#
# Argyris is defined on triangles, so we simplexify a Cartesian mesh.

n = 8
model = simplexify(CartesianDiscreteModel((0, 1, 0, 1), (n, n)))

# ## Manufactured solution
#
# We take $u = p(x)p(y)$ with $p(t) = t^2(1-t)^2$, which vanishes together with
# its gradient on $\partial\Omega$, and compute $f = \Delta^2 u$ by hand.

p(t) = t^2 * (1 - t)^2
pdd(t) = 2 - 12*t + 12*t^2

uex(x) = p(x[1]) * p(x[2])
f(x) = 24*p(x[2]) + 2*pdd(x[1]) * pdd(x[2]) + 24*p(x[1])

# ## FE space
#
# `argyris` is a scalar element of order 5 on triangles. The boundary degrees of
# freedom include the vertex Hessians, and $D^2 u \neq 0$ on $\partial\Omega$
# even for a clamped plate, so the Dirichlet data is the exact solution: its
# interpolation supplies the right value for every boundary degree of freedom at
# once.

V = FESpace(model, ReferenceFE(argyris, Float64, 5); dirichlet_tags="boundary")
U = TrialFESpace(V, uex)

Ω = Triangulation(model)
dΩ = Measure(Ω, 12)

a(u, v) = ∫( ∇∇(u) ⊙ ∇∇(v) )dΩ
l(v) = ∫( f * v )dΩ

op = AffineFEOperator(a, l, U, V)
uh = solve(op)

# ## Errors
#
# The space is $P_5$, so we expect $O(h^6)$ in $L^2$ and $O(h^4)$ in the $H^2$
# seminorm.

Hex = ∇∇(uex)

e = uh - uex
eH = ∇∇(uh) - Hex
l2 = sqrt(sum( ∫( e * e )dΩ ))
h2 = sqrt(sum( ∫( eH ⊙ eH )dΩ ))
println("L2 error: ", l2)
println("H2 seminorm error: ", h2)

# ## The space really is $C^1$
#
# Both the value and the *full* gradient are continuous across every interior
# edge — not just the mean normal derivative, as in Morley. We check on a
# function that is not in the space, so that the jumps are not zero for the
# trivial reason. Note the space here carries no `dirichlet_tags`: interpolating
# into a constrained space would zero the boundary degrees of freedom and inflate
# the jumps near $\partial\Omega$.

w(x) = sin(2*x[1]) * cos(3*x[2])
wh = interpolate(w, FESpace(model, ReferenceFE(argyris, Float64, 5)))

Λ = SkeletonTriangulation(model)
dΛ = Measure(Λ, 12)

println("value jump:    ", sqrt(sum( ∫( jump(wh) * jump(wh) )dΛ )))
println("gradient jump: ", sqrt(sum( ∫( jump(∇(wh)) ⋅ jump(∇(wh)) )dΛ )))

# ... but not $C^2$: the Hessian does jump, as it must for a finite element space
# built on a triangulation.

println("Hessian jump:  ", sqrt(sum( ∫( jump(∇∇(wh)) ⊙ jump(∇∇(wh)) )dΛ )))

# ## Visualisation

mkpath("output_path")
writevtk(Ω, "output_path/argyris", cellfields=["u" => uh, "error" => e])
