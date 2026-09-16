# In this tutorial, we will learn
#  - How to solve linear elasticity in mixed form with *weakly* imposed symmetry
#  - How to build a tensor-valued $H(\mathrm{div})$ space by stacking copies of a
#    vector-valued element with `CartProdRefFE`
#  - Why the copies are stacked on the *last* index, and what that changes in the
#    weak form
#
# ## Problem statement
#
# The Arnold--Winther tutorial solves the Hellinger--Reissner system with
# a stress space that is symmetric pointwise. Building such a space is hard. The
# alternative is to drop the symmetry from the space and impose it with a
# Lagrange multiplier $\gamma$, the *rotation*. With the identity compliance the
# system is: find $(\sigma, u, \gamma)$ with
#
# ```math
# \begin{aligned}
# \int_\Omega \sigma : \tau + \int_\Omega u\cdot\operatorname{div}\tau
#   + \int_\Omega \gamma\, (\chi : \tau) &= 0, \\
# \int_\Omega v\cdot\operatorname{div}\sigma &= -\int_\Omega f\cdot v, \\
# \int_\Omega \eta\, (\chi : \sigma) &= 0,
# \end{aligned}
# ```
#
# for all $(\tau, v, \eta)$, where
#
# ```math
# \chi = \begin{pmatrix} 0 & 1 \\ -1 & 0\end{pmatrix},
# \qquad \chi : \tau = \tau_{12} - \tau_{21},
# ```
#
# so the third equation says exactly that $\sigma$ is symmetric in the weak
# sense, and $\gamma$ converges to $\tfrac12(\partial_1 u_2 - \partial_2 u_1)$.
# The stress is now an arbitrary tensor field with rows -- or columns, see below
# -- in $H(\operatorname{div})$, which is a space we already have.
#
# ## The stress space
#
# `CartProdRefFE(reffe, K)` stacks `K` independent copies of any reference
# element. With a vector-valued atom the values become tensors,
#
# ```math
# \texttt{VectorValue\{d\}} \;\longmapsto\; \texttt{TensorValue\{d,K\}},
# ```
#
# one copy per **column**. Two copies of a $2$-dimensional $H(\operatorname{div})$
# element therefore give a tensor field whose every column is
# $H(\operatorname{div})$ -- the stress space of elasticity with weakly imposed
# symmetry.
#
# The copies go on the *last* index because Gridap writes the derivative index
# *first*, $(\nabla A)_{kij} = \partial_k A_{ij}$. Appending the copy index is
# then the one placement that commutes with differentiation, and one consequence
# is exactly what this tutorial needs: since
# $\operatorname{div} A = \operatorname{tr}(\nabla A)$ and `tr` of a third-order
# tensor traces its first two indices,
#
# ```math
# (\operatorname{div} A)_j = \partial_i A_{ij},
# ```
#
# so `∇⋅S` on a stacked space is precisely the vector of the copies'
# divergences. The literature writes this element row-wise instead; the system
# above is its transpose, and since the exact stress is symmetric the two
# describe the same solution.
#
# ## The elements
#
# Arnold, Falk and Winther's family on simplices takes, for $k \ge 1$,
#
# ```math
# \Sigma_h = \mathrm{BDM}_k^{\,2}, \qquad
# V_h = P_{k-1}^{\mathrm{dc}}(\mathbb{R}^2), \qquad
# Q_h = P_{k-1}^{\mathrm{dc}},
# ```
#
# and converges at $O(h^k)$ in all three unknowns. `CartProdRefFE` builds
# $\Sigma_h$ from the `bdm` element already in Gridap.

using Gridap
using Gridap.TensorValues
using Gridap.MultiField

# ## Discrete model

n = 8
model = simplexify(CartesianDiscreteModel((0, 1, 0, 1), (n, n)))

Ω = Triangulation(model)
degree = 8
dΩ = Measure(Ω, degree)

# ## Manufactured solution
#
# The same displacement as the Arnold--Winther tutorial, vanishing on
# $\partial\Omega$, together with its stress $\sigma = \varepsilon(u)$, its
# rotation $\gamma$, and the body force. Note $\sigma$ is written as a full
# `TensorValue`: the discrete stress is not symmetric, so the error cannot be
# measured against a `SymTensorValue`.

const π2 = pi^2
uex(x) = VectorValue(sin(pi*x[1]) * sin(pi*x[2]), sin(2*pi*x[1]) * sin(pi*x[2]))

function σex(x)
  u1x = pi * cos(pi*x[1]) * sin(pi*x[2])
  u1y = pi * sin(pi*x[1]) * cos(pi*x[2])
  u2x = 2*pi * cos(2*pi*x[1]) * sin(pi*x[2])
  u2y = pi * sin(2*pi*x[1]) * cos(pi*x[2])
  s12 = (u1y + u2x) / 2
  TensorValue{2,2,Float64}(u1x, s12, s12, u2y)
end

function γex(x)
  u1y = pi * sin(pi*x[1]) * cos(pi*x[2])
  u2x = 2*pi * cos(2*pi*x[1]) * sin(pi*x[2])
  (u2x - u1y) / 2
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
# The three spaces, and the system written exactly as above. The displacement
# boundary condition $u = 0$ is natural here, so no `dirichlet_tags` appear, and
# no integration by parts is needed: `∇⋅S` is available directly on the stacked
# space.

const χ = TensorValue{2,2,Float64}(0.0, -1.0, 1.0, 0.0)

function solve_weak_symmetry(k)
  Σ = FESpace(model, CartProdRefFE(ReferenceFE(TRI, bdm, Float64, k), 2))
  V = FESpace(model, ReferenceFE(lagrangian, VectorValue{2,Float64}, k-1); conformity=:L2)
  Q = FESpace(model, ReferenceFE(lagrangian, Float64, k-1); conformity=:L2)
  X = MultiFieldFESpace([Σ, V, Q])
  Y = MultiFieldFESpace([Σ, V, Q])

  a((S, u, g), (T, v, e)) = ∫( S ⊙ T + u ⋅ (∇⋅T) + g * (χ ⊙ T) +
                                        v ⋅ (∇⋅S) + e * (χ ⊙ S) )dΩ
  l((T, v, e)) = ∫( -(fex ⋅ v) )dΩ

  solve(AffineFEOperator(a, l, X, Y))
end

S1, u1, γ1 = solve_weak_symmetry(1)
S2, u2, γ2 = solve_weak_symmetry(2)

# ## Errors
#
# $O(h^k)$ in all three unknowns, so the $k=2$ element is one order better
# everywhere.

for (k, Sh, uh, gh) in ((1, S1, u1, γ1), (2, S2, u2, γ2))
  eS = Sh - σex
  eu = uh - uex
  eg = gh - γex
  println("k = ", k,
          "  |sigma| = ", sqrt(sum( ∫( eS ⊙ eS )dΩ )),
          "  |u| = ",     sqrt(sum( ∫( eu ⋅ eu )dΩ )),
          "  |gamma| = ", sqrt(sum( ∫( eg * eg )dΩ )))
end

# ## How symmetric is it?
#
# Not pointwise, which is the price of the method: $\chi : \sigma_h$ is zero only
# against the multiplier space. Its $L^2$ norm converges to zero at the rate of
# the method, unlike for `aw_c`, where it is zero to round-off.

for (k, Sh) in ((1, S1), (2, S2))
  asym = χ ⊙ Sh
  println("k = ", k, "  |skew part| = ", sqrt(sum( ∫( asym * asym )dΩ )))
end

# ## Visualisation

mkpath("output_path")
writevtk(Ω, "output_path/weak_symmetry_elasticity",
         cellfields=["sigma" => S2, "u" => u2, "gamma" => γ2])
