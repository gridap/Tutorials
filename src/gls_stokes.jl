# In this tutorial, we will learn
#  - How to use the Gopalakrishnan--Lederer--Schöberl element, a
#    *traceless*-matrix-valued element that is normal--tangential continuous
#  - How to reconstruct a velocity gradient in it
#  - Where it fits in the mass-conserving mixed stress formulation of Stokes
#
# ## The element
#
# The **GLS** element (of the second kind) is the space of *traceless* matrix
# fields
#
# ```math
# GLS_r(K) = P_r(K;\mathbb{M}_0), \qquad \mathbb{M}_0 = \{M : \operatorname{tr} M = 0\},
# ```
#
# of dimension $3(r+1)(r+2)/2$ for any $r \ge 0$, with degrees of freedom
#
# ```math
# \int_e (t\cdot M n)\,\mu_i \ \text{ on each edge}, \qquad
# \int_K (t_e\cdot M n_e)\, q \ \text{ in the cell},
# ```
#
# both of them the *same* normal--tangential functional — over the edges, which
# glue, and over the cell, which does not. Only $t\cdot M n$ is shared between two
# triangles.
#
# It completes a small family. Three elements share exactly the same shape of
# change of basis, because each pairs its matrix with two directions carried
# dually by its own push-forward:
#
# | element | degree of freedom | continuity |
# |---|---|---|
# | `hhj`   | $n\cdot M n$ | normal--normal |
# | `regge` | $t\cdot M t$ | tangential--tangential |
# | `gls`   | $t\cdot M n$ | normal--tangential |
#
# ## Why traceless
#
# In the mass-conserving mixed stress (MCS) formulation of Stokes flow, the
# unknowns are the velocity $u$ in an $H(\operatorname{div})$-conforming space,
# the pressure, and the *deviatoric* velocity gradient
# $\sigma = \nabla u - \tfrac{1}{2}(\operatorname{div} u) I$. Since
# $\operatorname{div} u = 0$ is imposed exactly by the velocity space, $\sigma$ is
# traceless — and only its normal--tangential component needs to be
# single-valued, because that is all the formulation tests. GLS is the space that
# supplies exactly that and no more.
#
# Note the element is built as a *constrained* subspace of the full matrix space
# $P_r(K;\mathbb{M})$, so `length(get_prebasis(reffe))` exceeds `num_dofs(reffe)`
# by the number of trace constraints.

using Gridap
using Gridap.TensorValues
using Gridap.ReferenceFEs

# ## Discrete model and space

n = 8
model = simplexify(CartesianDiscreteModel((0, 1, 0, 1), (n, n)))

Ω = Triangulation(model)
dΩ = Measure(Ω, 8)

r = 1
V = FESpace(model, ReferenceFE(gls, Float64, r))

reffe = ReferenceFE(TRI, gls, Float64, r)
println("dofs per cell: ", num_dofs(reffe), "   ambient prebasis: ", length(get_prebasis(reffe)))

# ## Reconstructing a deviatoric gradient
#
# Take a divergence-free velocity and project its gradient into the GLS space.
# Because $\operatorname{div} u = 0$, that gradient is already traceless, so it
# lies in the space as soon as it is a polynomial of degree $\le r$ — and then
# the projection is exact. Here $u = (y^2, -x^2)$, whose gradient is linear.

u(x) = VectorValue(x[2]^2, -x[1]^2)
G = ∇(u)

a(σ, τ) = ∫( σ ⊙ τ )dΩ
l(τ) = ∫( G ⊙ τ )dΩ
σh = solve(AffineFEOperator(a, l, V, V))

e = σh - G
println("L2 error of the projected gradient: ", sqrt(sum( ∫( e ⊙ e )dΩ )))

# For a velocity whose gradient is *not* in the space, the projection is the best
# approximation in $L^2$ rather than the field itself:

ψ(x) = (x[1]^2 * (1 - x[1])^2) * (x[2]^2 * (1 - x[2])^2)
v(x) = VectorValue(∇(ψ)(x)[2], -∇(ψ)(x)[1])
Gv = ∇(v)
σv = solve(AffineFEOperator(a, τ -> ∫( Gv ⊙ τ )dΩ, V, V))
ev = σv - Gv
println("L2 error, gradient outside the space: ", sqrt(sum( ∫( ev ⊙ ev )dΩ )))

# The reconstruction really is traceless, pointwise:

tr_σ = Operation(tr)(σh)
println("L2 norm of the trace: ", sqrt(sum( ∫( tr_σ * tr_σ )dΩ )))

# ## Normal--tangential continuity, and nothing more
#
# The defining property. As before we interpolate a field that is not in the
# space, so the full jump is not zero for a trivial reason.

g(x) = TensorValue(sin(2*x[1]), cos(3*x[2]), sin(x[1] + x[2]), -sin(2*x[1]))
gh = interpolate(g, V)

Λ = SkeletonTriangulation(model)
dΛ = Measure(Λ, 8)
nΛ = get_normal_vector(Λ).⁺
tΛ = Operation(n -> VectorValue(n[2], -n[1]))(nΛ)

nt = sqrt(sum( ∫( (tΛ ⋅ jump(gh) ⋅ nΛ) * (tΛ ⋅ jump(gh) ⋅ nΛ) )dΛ ))
full = sqrt(sum( ∫( jump(gh) ⊙ jump(gh) )dΛ ))
println("jump of the nt component: ", nt)
println("jump of the full matrix:  ", full)

# ## The family

for rr in 0:2
  Vr = FESpace(model, ReferenceFE(gls, Float64, rr))
  println("r = ", rr, "   dofs = ", num_free_dofs(Vr))
end

# ## Visualisation

mkpath("output_path")
writevtk(Ω, "output_path/gls", cellfields=["sigma" => σh])
