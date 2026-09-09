# In this tutorial, we will learn
#  - How to use the Regge element, a symmetric-tensor-valued element that is
#    *tangential--tangential* continuous
#  - What its degrees of freedom mean geometrically
#  - How it mirrors the Hellan--Herrmann--Johnson element
#
# ## The element
#
# The **Regge** element is the full symmetric-matrix-valued polynomial space
# $P_r(K;\mathbb{S})$, of any degree $r \ge 0$, with degrees of freedom
#
# ```math
# \int_e (t\cdot M t)\,\mu_i \ \text{ on each edge}, \qquad
# \int_K M : \tau \ \text{ in the cell},
# ```
#
# $t$ the unit tangent of the edge. Only the **tangential--tangential** component
# is shared between two triangles; everything else jumps.
#
# It is the exact mirror of the Hellan--Herrmann--Johnson element, which shares
# $n\cdot\sigma n$ instead — same space, same interior moments, same dimension,
# the normal swapped for the tangent. Their push-forwards differ accordingly
# (double covariant rather than double contravariant Piola), and both end up with
# the same, diagonal, change of basis.
#
# ## What the degrees of freedom mean
#
# Regge elements come from Regge calculus, where a geometry is described not by
# coordinates but by **edge lengths**. That is exactly what the lowest-order
# degrees of freedom are. If $M$ is a perturbation of the metric, the induced
# change in the squared length of an edge $e$ with tangent $t$ and length $|e|$ is
#
# ```math
# \delta(|e|^2) = |e| \int_e t\cdot M t ,
# ```
#
# i.e. the Regge degree of freedom, up to the edge length. A discrete metric in
# the lowest-order Regge space is *precisely* an assignment of one number per
# edge — which is why the tangential--tangential component, and only that, has to
# be single-valued: two triangles sharing an edge must agree on its length.

using Gridap
using Gridap.TensorValues
using Gridap.Geometry

# ## Discrete model

n = 8
model = simplexify(CartesianDiscreteModel((0, 1, 0, 1), (n, n)))

Ω = Triangulation(model)
dΩ = Measure(Ω, 6)

# ## The space
#
# `regge` is a symmetric-tensor-valued element of degree `r`. At `r = 0` it has
# exactly one degree of freedom per edge and none in the cell.

r = 0
V = FESpace(model, ReferenceFE(regge, Float64, r))

topo = get_grid_topology(model)
println("dofs: ", num_free_dofs(V), "   edges: ", num_faces(topo, 1))

# ## Interpolation
#
# The space is all of $P_r(K;\mathbb{S})$, so it reproduces any symmetric tensor
# field of that degree exactly. Here `r = 0`: constants.

Mconst(x) = SymTensorValue{2,Float64}(2.0, -1.0, 3.0)
Mh = interpolate(Mconst, V)
e = Mh - Mconst
println("interpolation error, constant metric: ", sqrt(sum( ∫( e ⊙ e )dΩ )))

# ## Tangential--tangential continuity
#
# The defining property. We interpolate a field that is *not* in the space, so
# the jumps are not zero for a trivial reason, and measure the jump of
# $t\cdot M t$ against the jump of the whole tensor.

g(x) = SymTensorValue{2,Float64}(sin(2*x[1]), cos(3*x[2]), sin(x[1] + x[2]))
gh = interpolate(g, V)

Λ = SkeletonTriangulation(model)
dΛ = Measure(Λ, 6)
nΛ = get_normal_vector(Λ).⁺
tΛ = Operation(n -> VectorValue(n[2], -n[1]))(nΛ)

tt = sqrt(sum( ∫( (tΛ ⋅ jump(gh) ⋅ tΛ) * (tΛ ⋅ jump(gh) ⋅ tΛ) )dΛ ))
full = sqrt(sum( ∫( jump(gh) ⊙ jump(gh) )dΛ ))
println("jump of the tt component: ", tt)
println("jump of the full tensor:  ", full)

# Compare with the mirror element, which shares the normal--normal component
# instead — on the same mesh, with the same field, the two continuities are
# exchanged:

Vh = FESpace(model, ReferenceFE(hhj, Float64, r))
hh = interpolate(g, Vh)
println("HHJ, jump of the nn component: ",
        sqrt(sum( ∫( (nΛ ⋅ jump(hh) ⋅ nΛ) * (nΛ ⋅ jump(hh) ⋅ nΛ) )dΛ )))
println("HHJ, jump of the tt component: ",
        sqrt(sum( ∫( (tΛ ⋅ jump(hh) ⋅ tΛ) * (tΛ ⋅ jump(hh) ⋅ tΛ) )dΛ )))

# ## Higher degree
#
# The element is a family. At degree `r` there are `r+1` degrees of freedom per
# edge and `3r(r+1)/2` in the cell, and the space reproduces any
# $P_r(K;\mathbb{S})$ field exactly.

for r in 0:2
  Vr = FESpace(model, ReferenceFE(regge, Float64, r))
  Mr(x) = SymTensorValue{2,Float64}(1.0 + x[1]^r, 2.0 - x[2]^r, 3.0 + (x[1]*x[2])^(r ÷ 2))
  er = interpolate(Mr, Vr) - Mr
  println("r = ", r, "   dofs = ", num_free_dofs(Vr),
          "   interpolation error = ", sqrt(sum( ∫( er ⊙ er )dΩ )))
end

# ## Visualisation

mkpath("output_path")
writevtk(Ω, "output_path/regge", cellfields=["M" => gh])
