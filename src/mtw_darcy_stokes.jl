# In this tutorial, we will learn
#  - How to discretise a problem that interpolates between Darcy and Stokes flow
#  - How to use the Mardal--Tai--Winther element, which is stable uniformly in
#    the parameter
#  - Why an element can be $H(\operatorname{div})$-conforming and only *weakly*
#    $H^1$
#
# ## Problem statement
#
# The Darcy--Stokes (or Brinkman) problem: find the velocity $u$ and the pressure
# $p$ with
#
# ```math
# u - \varepsilon^2 \Delta u + \nabla p = f, \qquad \operatorname{div} u = 0,
# ```
#
# on $\Omega = (0,1)^2$. At $\varepsilon \sim 1$ this is Stokes flow; as
# $\varepsilon \to 0$ it degenerates to Darcy flow. A discretisation that is
# stable for *both* limits, and uniformly in between, is the hard part: a Stokes
# pair typically degrades as $\varepsilon \to 0$, and a Darcy pair
# ($H(\operatorname{div})$-conforming, discontinuous tangentially) cannot control
# the viscous term.
#
# ## The element
#
# The **Mardal--Tai--Winther** element resolves this. On a triangle it is the
# 9-dimensional space
#
# ```math
# MTW(K) = P_1(K;\mathbb{R}^2) + \operatorname{curl}(b\,P_1(K)),
# ```
#
# $b$ the cubic bubble, with three degrees of freedom per edge: two moments of
# $u\cdot n$ against $P_1(e)$, and one of $u\cdot t$ against the constant.
#
# Sharing the normal moments makes the space $H(\operatorname{div})$-conforming,
# so `div u = 0` is meaningful and the Darcy limit is fine. Sharing the single
# tangential moment makes it *weakly* $H^1$: the tangential trace jumps, but its
# mean over each edge does not, and that is enough to control the viscous term
# uniformly in $\varepsilon$.
#
# The same element exists on tetrahedra, where it is the element of Tai and
# Winther, with six degrees of freedom per face; `mtw` covers both.

using Gridap
using Gridap.MultiField

# ## Discrete model
#
# MTW is defined on simplices.

n = 16
model = simplexify(CartesianDiscreteModel((0, 1, 0, 1), (n, n)))

Ω = Triangulation(model)
dΩ = Measure(Ω, 8)

# ## Manufactured solution
#
# The velocity is a curl, so it is divergence free by construction, and the
# stream function vanishes to second order on $\partial\Omega$, so $u$ vanishes
# there too.

ψ(x) = (x[1]^2 * (1 - x[1])^2) * (x[2]^2 * (1 - x[2])^2)
uex(x) = VectorValue(∇(ψ)(x)[2], -∇(ψ)(x)[1])
pex(x) = cos(pi * x[1]) * cos(pi * x[2])

# ## The solver
#
# The pressure is piecewise constant, with the mean fixed. Note `∇(u)` here is
# the *broken* gradient: the space is not $H^1$, and the viscous term is
# assembled element by element.

function solve_darcy_stokes(ε)
  f(x) = uex(x) - ε^2 * Δ(uex)(x) + ∇(pex)(x)

  V = FESpace(model, ReferenceFE(mtw, Float64, 1); dirichlet_tags="boundary")
  Q = FESpace(model, ReferenceFE(lagrangian, Float64, 0);
              conformity=:L2, constraint=:zeromean)
  X = MultiFieldFESpace([TrialFESpace(V, uex), TrialFESpace(Q)])
  Y = MultiFieldFESpace([V, Q])

  a((u, p), (v, q)) = ∫( u ⋅ v + ε^2 * (∇(u) ⊙ ∇(v)) - p * (∇ ⋅ v) - q * (∇ ⋅ u) )dΩ
  l((v, q)) = ∫( f ⋅ v )dΩ

  solve(AffineFEOperator(a, l, X, Y))
end

# ## Uniform robustness
#
# The point of the element: the velocity error barely moves as $\varepsilon$
# sweeps four orders of magnitude, from the Stokes regime to the Darcy one.

for ε in (1.0, 1e-1, 1e-2, 1e-4)
  uh, ph = solve_darcy_stokes(ε)
  eu = uh - uex
  ep = ph - pex
  println("eps = ", ε,
          "   |u| = ", sqrt(sum( ∫( eu ⋅ eu )dΩ )),
          "   |p| = ", sqrt(sum( ∫( ep * ep )dΩ )))
end

# ## H(div)-conforming, weakly $H^1$
#
# Interpolating a field outside the space shows both halves of the statement. The
# normal component is continuous outright; the tangential component jumps, but
# its mean over each edge is zero.

w(x) = VectorValue(sin(3*x[1]) * x[2]^2, cos(2*x[2]) + x[1]^3)
V = FESpace(model, ReferenceFE(mtw, Float64, 1))
wh = interpolate(w, V)

Λ = SkeletonTriangulation(model)
dΛ = Measure(Λ, 8)
nΛ = get_normal_vector(Λ)
tΛ = Operation(n -> VectorValue(n[2], -n[1]))(nΛ)

println("normal jump:            ", sqrt(sum( ∫( jump(wh ⋅ nΛ) * jump(wh ⋅ nΛ) )dΛ )))
println("tangential jump:        ", sqrt(sum( ∫( jump(wh ⋅ tΛ) * jump(wh ⋅ tΛ) )dΛ )))
println("mean tangential jump:   ", maximum(abs, get_array( ∫( jump(wh ⋅ tΛ) )dΛ )))

# ## Visualisation

uh, ph = solve_darcy_stokes(1e-4)
mkpath("output_path")
writevtk(Ω, "output_path/mtw", cellfields=["u" => uh, "p" => ph])
