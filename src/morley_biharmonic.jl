# In this tutorial, we will learn
#  - How to solve a fourth-order (biharmonic) problem in Gridap
#  - How to use the Morley element, a *nonconforming* plate element
#  - How to read an unstructured triangular mesh generated with Gmsh
#
# ## Problem statement
#
# We solve the **clamped plate**: find the transverse deflection $w$ of a thin
# plate $\Omega$ loaded by $f$ and clamped along its whole boundary,
#
# ```math
# \Delta^2 w = f \ \text{ in } \Omega, \qquad
# w = \frac{\partial w}{\partial n} = 0 \ \text{ on } \partial\Omega .
# ```
#
# We take $\Omega$ the unit disk and $f \equiv 1$, for which the deflection is
# radially symmetric and known in closed form,
#
# ```math
# w(r) = \frac{(1-r^2)^2}{64},
# ```
# so the centre deflection is exactly $1/64$.
#
# ## Why a special element
#
# The natural variational form of the problem lives in $H^2(\Omega)$,
#
# ```math
# a(w,v) = \int_\Omega D^2 w : D^2 v, \qquad \ell(v) = \int_\Omega f\, v,
# ```
#
# and an $H^2$-conforming space must be $C^1$ across element boundaries. That is
# expensive: the cheapest $C^1$ triangle is the quintic Argyris element, with 21
# degrees of freedom (see its own tutorial).
#
# The **Morley element** buys simplicity by giving up conformity. It is the
# quadratic space $P_2(K)$ with only six degrees of freedom per triangle,
#
# ```math
# \ell^v(w) = w(v) \ \text{ at each vertex}, \qquad
# \ell^e(w) = \int_e \nabla w\cdot n \ \text{ on each edge},
# ```
#
# so neither $w$ nor $\nabla w$ is continuous across an edge — only the vertex
# values and the *mean* normal derivative are shared. The bilinear form is then
# applied element by element (a *broken* Hessian), and the method still converges.
# This is the classical nonconforming plate element of Morley.
#
# ## Discrete model
#
# We read an unstructured triangular mesh of the unit disk. The mesh is generated
# by `assets/nonstandard_reffes/MeshGenerator.jl` and carries a physical group
# named `"boundary"` on the rim.

using Gridap
using GridapGmsh

model = GmshDiscreteModel("../models/clamped_disk.msh")

# ## FE space
#
# `morley` is a scalar element of order 2 on triangles. Its conformity is
# `H1Conformity()`, which in Gridap only says *which faces own and share degrees
# of freedom* — here vertices and edges. It is not a claim that the space is
# $C^0$; Morley is neither $C^0$ nor $C^1$.
#
# Both Morley degrees of freedom on the boundary — the vertex value and the mean
# normal derivative — are exactly the two clamped conditions, so the boundary
# condition is homogeneous.

V = FESpace(model, ReferenceFE(morley, Float64, 2); dirichlet_tags="boundary")
U = TrialFESpace(V)

# ## Weak form and solve
#
# `∇∇(u)` is the element-wise Hessian. On a conforming space it would be the
# distributional one; here the two differ, which is exactly the nonconformity.

Ω = Triangulation(model)
dΩ = Measure(Ω, 6)

f(x) = 1.0

a(u, v) = ∫( ∇∇(u) ⊙ ∇∇(v) )dΩ
l(v) = ∫( f * v )dΩ

op = AffineFEOperator(a, l, U, V)
wh = solve(op)

# ## Checking the result
#
# The exact deflection is $(1-r^2)^2/64$. We compare in the $L^2$ norm. The
# domain is meshed by straight-edged triangles, so the geometry itself is only
# approximated and the error will not go below that; on this mesh it is around a
# percent of the peak deflection.

wex(x) = (1 - (x[1]^2 + x[2]^2))^2 / 64

wexh = CellField(wex, Ω)
e = wh - wexh
l2 = sqrt(sum( ∫( e * e )dΩ ))
peak = sqrt(sum( ∫( wexh * wexh )dΩ ))
println("relative L2 error: ", l2 / peak)

# The deflection at the centre should be close to $1/64 = 0.015625$:

println("exact centre deflection:  ", 1 / 64)

# ## Visualisation
#
# We write the deflection and the broken Hessian, whose jump across edges is what
# distinguishes a nonconforming method from a conforming one.

mkpath("output_path")
writevtk(Ω, "output_path/morley", cellfields=["w" => wh, "hessian" => ∇∇(wh)])

# ## Nonconformity, made visible
#
# The value and the normal derivative do not match across an edge, but the *mean*
# of the normal derivative does — that is the shared degree of freedom. Averaged
# over each interior edge, the jump is at round-off:

Λ = SkeletonTriangulation(model)
dΛ = Measure(Λ, 6)
n = get_normal_vector(Λ)

facet_means = get_array( ∫( jump(∇(wh) ⋅ n) )dΛ )
println("largest facet-mean jump of ∂w/∂n: ", maximum(abs, facet_means))
