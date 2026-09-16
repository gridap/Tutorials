# In this tutorial, we will learn
#  - How to solve linear elasticity in *mixed* (Hellinger--Reissner) form
#  - How to use the Arnold--Winther elements, which impose the symmetry of the
#    stress tensor exactly
#  - What the conforming and nonconforming variants buy you
#
# ## Problem statement
#
# The Hellinger--Reissner formulation of linear elasticity takes the stress
# $\sigma$ and the displacement $u$ as unknowns: with the identity compliance,
#
# ```math
# \int_\Omega \sigma : \tau + b(\tau, u) = 0, \qquad
# b(\sigma, v) = -\int_\Omega f\cdot v, \qquad
# b(\tau, v) = \sum_K \int_K \operatorname{div}\tau \cdot v .
# ```
#
# Its difficulty is well known: a stable pair needs a stress space that is
# **symmetric**, **$H(\operatorname{div})$-conforming** and matched to the
# displacement space. Elements satisfying all three eluded the field for decades.
#
# ## The elements
#
# Arnold and Winther gave the first two on triangles.
#
# The **conforming** element, `aw_c`, has 24 degrees of freedom:
#
# ```math
# AW_c(K) = \{\tau \in P_3(K;\mathbb{S}) : \operatorname{div}\tau \in P_1(K;\mathbb{R}^2)\},
# ```
#
# with the three components of $\tau$ at each vertex, the degree 0 and 1 moments
# of $n\cdot\tau n$ and $n\cdot\tau t$ on each edge, and three interior moments.
# It is genuinely $H(\operatorname{div};\mathbb{S})$-conforming: the full traction
# $\tau\cdot n$ is continuous.
#
# The **nonconforming** element, `aw_nc`, has 15:
#
# ```math
# AW_{nc}(K) = \{\tau \in P_2(K;\mathbb{S}) : (n\cdot\tau n)|_e \in P_1(e)\},
# ```
#
# with only the edge moments and three interior ones. The traction jumps, but the
# jump is orthogonal to the $P_1$ displacement space, which is enough for
# convergence — the standard trade of accuracy for cost.
#
# Both are *constrained* subspaces of a larger polynomial space, so for them
# `length(get_prebasis(reffe))` is larger than `num_dofs(reffe)`. That is by
# design and nothing downstream depends on the two being equal.

using Gridap
using Gridap.TensorValues
using Gridap.MultiField

# ## Discrete model

n = 8
model = simplexify(CartesianDiscreteModel((0, 1, 0, 1), (n, n)))

Ω = Triangulation(model)
Λ = SkeletonTriangulation(model)
Γ = BoundaryTriangulation(model)
degree = 8
dΩ = Measure(Ω, degree)
dΛ = Measure(Λ, degree)
dΓ = Measure(Γ, degree)
nΛ = get_normal_vector(Λ)
nΓ = get_normal_vector(Γ)

# ## Manufactured solution
#
# A displacement vanishing on $\partial\Omega$, its symmetric gradient, and the
# body force that produces it.

const π2 = pi^2
uex(x) = VectorValue(sin(pi*x[1]) * sin(pi*x[2]), sin(2*pi*x[1]) * sin(pi*x[2]))

function σex(x)
  u1x = pi * cos(pi*x[1]) * sin(pi*x[2])
  u1y = pi * sin(pi*x[1]) * cos(pi*x[2])
  u2x = 2*pi * cos(2*pi*x[1]) * sin(pi*x[2])
  u2y = pi * sin(2*pi*x[1]) * cos(pi*x[2])
  SymTensorValue{2,Float64}(u1x, (u1y + u2x) / 2, u2y)
end

function fex(x)
  u1 = sin(pi*x[1]) * sin(pi*x[2])
  u2 = sin(2*pi*x[1]) * sin(pi*x[2])
  d11u1, d22u1 = -π2*u1, -π2*u1
  d12u1 = π2 * cos(pi*x[1]) * cos(pi*x[2])
  d11u2, d22u2 = -4*π2*u2, -π2*u2
  d12u2 = 2*π2 * cos(2*pi*x[1]) * cos(pi*x[2])
  VectorValue(-(d11u1 + (d22u1 + d12u2)/2), -((d12u1 + d11u2)/2 + d22u2))
end

# ## The solver
#
# The displacement space is discontinuous vector $P_1$. The bilinear form $b$ is
# integrated by parts element by element, so no derivative of $\tau$ is ever
# taken and each side of a facet contributes with its own traction — which is
# what keeps the form consistent for the *nonconforming* stress space. The
# displacement boundary condition $u = 0$ is natural here, so no
# `dirichlet_tags` appear.

function solve_elasticity(reffe_σ)
  Vσ = FESpace(model, reffe_σ)
  Vu = FESpace(model, ReferenceFE(lagrangian, VectorValue{2,Float64}, 1); conformity=:L2)
  X = MultiFieldFESpace([Vσ, Vu])
  Y = MultiFieldFESpace([Vσ, Vu])

  b(τ, v) = ∫( -(τ ⊙ ε(v)) )dΩ +
            ∫( ((τ.⁺ ⋅ nΛ.⁺) ⋅ v.⁺) + ((τ.⁻ ⋅ nΛ.⁻) ⋅ v.⁻) )dΛ +
            ∫( ((τ ⋅ nΓ) ⋅ v) )dΓ
  A((σ, u), (τ, v)) = ∫( σ ⊙ τ )dΩ + b(τ, u) + b(σ, v)
  L((τ, v)) = ∫( -(fex ⋅ v) )dΩ

  solve(AffineFEOperator(A, L, X, Y))
end

σc, uc = solve_elasticity(ReferenceFE(TRI, aw_c, Float64))
σn, un = solve_elasticity(ReferenceFE(TRI, aw_nc, Float64))

# ## Errors
#
# The conforming element gives $O(h^3)$ in the stress and $O(h^2)$ in the
# displacement; the nonconforming one, being built on $P_2$ rather than $P_3$,
# gives $O(h)$ and $O(h^2)$.

for (name, σh, uh) in (("conforming   ", σc, uc), ("nonconforming", σn, un))
  eσ = σh - σex
  eu = uh - uex
  println(name, "  |sigma| = ", sqrt(sum( ∫( eσ ⊙ eσ )dΩ )),
                "   |u| = ",     sqrt(sum( ∫( eu ⋅ eu )dΩ )))
end

# ## Conforming or not
#
# The difference between the two spaces, made visible: interpolate a stress field
# that is in neither, and measure the jump of the traction $\tau\cdot n$ across
# the interior edges. For the conforming element it vanishes; for the
# nonconforming one it does not, though its normal--normal part still does.

g(x) = SymTensorValue{2,Float64}(sin(2*x[1]), cos(3*x[2]), sin(x[1] + x[2]))
nplus = get_normal_vector(Λ).⁺

for (name, reffe) in (("conforming   ", ReferenceFE(TRI, aw_c, Float64)),
                      ("nonconforming", ReferenceFE(TRI, aw_nc, Float64)))
  gh = interpolate(g, FESpace(model, reffe))
  jt = sqrt(sum( ∫( (jump(gh) ⋅ nplus) ⋅ (jump(gh) ⋅ nplus) )dΛ ))
  jnn = sqrt(sum( ∫( (nplus ⋅ jump(gh) ⋅ nplus) * (nplus ⋅ jump(gh) ⋅ nplus) )dΛ ))
  println(name, "  traction jump = ", jt, "   nn jump = ", jnn)
end

# ## Visualisation

mkpath("output_path")
writevtk(Ω, "output_path/arnold_winther",
         cellfields=["sigma_c" => σc, "u_c" => uc, "sigma_nc" => σn, "u_nc" => un])
