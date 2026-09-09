# In this tutorial, we will learn
#  - How to solve the Kirchhoff plate with a *mixed* method
#  - How to use the Hellan--Herrmann--Johnson element, a symmetric-tensor-valued
#    element that is only normal--normal continuous
#  - How to write a form that couples a cell integral to a skeleton integral
#
# ## Problem statement
#
# The clamped Kirchhoff plate again — $\Delta^2 w = f$ with $w = \partial_n w = 0$
# — but written as a first-order system in the bending moment tensor
# $\sigma = D^2 w$:
#
# ```math
# \sigma - D^2 w = 0, \qquad -\operatorname{div}\operatorname{div}\sigma = f .
# ```
#
# The advantage over the primal form is that the displacement space no longer
# needs to be $C^1$: continuous $P_{r+1}$ Lagrange suffices.
#
# ## The element
#
# The **Hellan--Herrmann--Johnson** element is the full symmetric-matrix-valued
# polynomial space $P_r(K;\mathbb{S})$, of any degree $r \ge 0$, with degrees of
# freedom
#
# ```math
# \int_e (n\cdot\sigma n)\,\mu_i \ \text{ on each edge}, \qquad
# \int_K \sigma : \tau \ \text{ in the cell}.
# ```
#
# Only the **normal--normal** component $n\cdot\sigma n$ is shared between two
# elements; the rest of the tensor jumps. That is exactly the regularity the
# mixed plate formulation needs, and no more — the trade-off that makes the
# method cheap.
#
# Because $n\cdot\sigma n$ is quadratic in $n$, the element needs no normal sign
# convention at all: reversing an edge leaves the functional unchanged.
#
# ## The mixed form
#
# Following Arnold and Walker, with $W_h$ the continuous Lagrange space of degree
# $r+1$,
#
# ```math
# b(\varphi, v) = -\sum_K \int_K \varphi : D^2 v
#               + \sum_e \int_e \varphi_{nn}\,[\![\partial_n v]\!],
# ```
#
# the discrete problem is: find $(\sigma_h, w_h)$ with
# $(\sigma_h,\tau) + b(\tau,w_h) = 0$ and $b(\sigma_h,v) = -\langle f,v\rangle$.
# Only $\varphi_{nn}$ appears in the skeleton term, which is precisely the
# component HHJ makes single-valued.

using Gridap
using Gridap.TensorValues
using Gridap.MultiField
using GridapGmsh

# ## Discrete model and spaces

n = 8
model = simplexify(CartesianDiscreteModel((0, 1, 0, 1), (n, n)))
r = 1

Ω = Triangulation(model)
Λ = SkeletonTriangulation(model)
Γ = BoundaryTriangulation(model)

degree = 2*r + 4
dΩ = Measure(Ω, degree)
dΛ = Measure(Λ, degree)
dΓ = Measure(Γ, degree)
nΛ = get_normal_vector(Λ).⁺
nΓ = get_normal_vector(Γ)

# `hhj` is a symmetric-tensor-valued element of degree `r`; the displacement uses
# ordinary Lagrange of degree `r+1`, with $w = 0$ imposed strongly. The other
# clamped condition, $\partial_n w = 0$, is imposed *weakly*, by including the
# boundary edges in the skeleton term with $[\![\partial_n v]\!] = \partial_n v$.

Vσ = FESpace(model, ReferenceFE(hhj, Float64, r))
Vw = FESpace(model, ReferenceFE(lagrangian, Float64, r + 1); dirichlet_tags="boundary")

X = MultiFieldFESpace([Vσ, TrialFESpace(Vw, 0.0)])
Y = MultiFieldFESpace([Vσ, Vw])

# ## Manufactured solution

p(t) = t^2 * (1 - t)^2
pdd(t) = 2 - 12*t + 12*t^2

wex(x) = p(x[1]) * p(x[2])
σex = ∇∇(wex)
f(x) = 24*p(x[2]) + 2*pdd(x[1]) * pdd(x[2]) + 24*p(x[1])

# ## Weak form and solve

bh(φ, v) = ∫( -(φ ⊙ ∇∇(v)) )dΩ +
           ∫( (nΛ ⋅ mean(φ) ⋅ nΛ) * (jump(∇(v)) ⋅ nΛ) )dΛ +
           ∫( (nΓ ⋅ φ ⋅ nΓ) * (∇(v) ⋅ nΓ) )dΓ

A((σ, w), (τ, v)) = ∫( σ ⊙ τ )dΩ + bh(τ, w) + bh(σ, v)
L((τ, v)) = ∫( -(f * v) )dΩ

σh, wh = solve(AffineFEOperator(A, L, X, Y))

# ## Errors
#
# The method converges at $O(h^{r+2})$ in the displacement and $O(h^{r+1})$ in
# the moment tensor.

ew = wh - wex
eσ = σh - σex
println("L2 error, displacement: ", sqrt(sum( ∫( ew * ew )dΩ )))
println("L2 error, moment:       ", sqrt(sum( ∫( eσ ⊙ eσ )dΩ )))

# ## Normal--normal continuity, and nothing more
#
# The defining property of the space. We interpolate a tensor field that is not
# in it, so that the full jump is not zero for a trivial reason: the
# normal--normal component is continuous to round-off, while the tensor itself
# jumps at $O(1)$.

g(x) = SymTensorValue{2,Float64}(sin(2*x[1]), cos(3*x[2]), sin(x[1] + x[2]))
gh = interpolate(g, Vσ)

nn = sqrt(sum( ∫( (nΛ ⋅ jump(gh) ⋅ nΛ) * (nΛ ⋅ jump(gh) ⋅ nΛ) )dΛ ))
full = sqrt(sum( ∫( jump(gh) ⊙ jump(gh) )dΛ ))
println("jump of the nn component: ", nn)
println("jump of the full tensor:  ", full)

# ## A plate with a re-entrant corner
#
# Nothing above used the structure of the square. The same driver runs on an
# unstructured mesh of the L-shaped domain, where the clamped plate has a genuine
# corner singularity and no closed-form solution — a uniform load is enough to
# see it.

model_L = GmshDiscreteModel("../models/lshape.msh")

Ω_L = Triangulation(model_L)
Λ_L = SkeletonTriangulation(model_L)
Γ_L = BoundaryTriangulation(model_L)
dΩ_L = Measure(Ω_L, degree)
dΛ_L = Measure(Λ_L, degree)
dΓ_L = Measure(Γ_L, degree)
nΛ_L = get_normal_vector(Λ_L).⁺
nΓ_L = get_normal_vector(Γ_L)

Vσ_L = FESpace(model_L, ReferenceFE(hhj, Float64, r))
Vw_L = FESpace(model_L, ReferenceFE(lagrangian, Float64, r + 1); dirichlet_tags="boundary")
X_L = MultiFieldFESpace([Vσ_L, TrialFESpace(Vw_L, 0.0)])
Y_L = MultiFieldFESpace([Vσ_L, Vw_L])

bh_L(φ, v) = ∫( -(φ ⊙ ∇∇(v)) )dΩ_L +
             ∫( (nΛ_L ⋅ mean(φ) ⋅ nΛ_L) * (jump(∇(v)) ⋅ nΛ_L) )dΛ_L +
             ∫( (nΓ_L ⋅ φ ⋅ nΓ_L) * (∇(v) ⋅ nΓ_L) )dΓ_L

A_L((σ, w), (τ, v)) = ∫( σ ⊙ τ )dΩ_L + bh_L(τ, w) + bh_L(σ, v)
L_L((τ, v)) = ∫( -(1.0 * v) )dΩ_L

σh_L, wh_L = solve(AffineFEOperator(A_L, L_L, X_L, Y_L))

mkpath("output_path")
writevtk(Ω, "output_path/hhj_square", cellfields=["w" => wh, "sigma" => σh])
writevtk(Ω_L, "output_path/hhj_lshape", cellfields=["w" => wh_L, "sigma" => σh_L])
