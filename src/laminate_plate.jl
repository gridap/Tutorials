# In this tutorial, we will learn
#  - How to build a single element whose components have *different*
#    conformities, with `CartProdRefFE`
#  - How to solve a plate problem in which stretching and bending are coupled,
#    using one displacement field rather than two unknowns
#  - How to write kinematics as slices of $\nabla U$ and $\nabla\nabla U$
#
# ## Problem statement
#
# A **laminated plate** is a stack of bonded layers. If the stack is symmetric
# about the mid-surface, stretching and bending do not interact and the two
# problems are solved independently — that is the usual situation, and it is why
# plate tutorials normally treat bending alone. If the stack is *unsymmetric*,
# with stiffer layers above than below, the two are coupled: pulling on it
# bends it, and bending it stretches it.
#
# Classical laminate theory writes this with three constitutive tensors, $A$ for
# stretching, $D$ for bending and $B$ for the coupling, $B$ vanishing exactly
# when the stack is symmetric. We take the simplest instance of it: all three
# proportional to the identity, with zero Poisson ratio, so that a single number
# $b$ carries the coupling. The unknowns are the in-plane displacement
# $u = (u_1,u_2)$ and the transverse deflection $w$, and the energy is
#
# ```math
# E(u,w) = \frac{1}{2}\int_\Omega
#   \varepsilon(u) : \varepsilon(u)
#   + 2b\, \varepsilon(u) : \kappa(w)
#   + \kappa(w) : \kappa(w)
#   \;-\; \int_\Omega f \cdot (u_1,u_2,w) ,
# ```
#
# with $\varepsilon(u)$ the symmetric gradient and $\kappa(w) = \nabla\nabla w$
# the curvature. The energy is positive definite for $|b| < 1$. At $b = 0$ it is
# the sum of a linear elasticity problem and a Kirchhoff plate problem.
#
# The two terms ask for different smoothness: $\varepsilon(u)$ needs
# $u \in H^1$, while $\kappa(w)$ needs $w \in H^2$. So the discrete space must
# be $C^0$ in its first two components and $C^1$ in the third.
#
# ## One field, not two
#
# Nothing *forces* the three components into one element — a `MultiFieldFESpace`
# pairing a vector Lagrange space with an Argyris space describes the same
# discrete problem. But these are not three unknowns that happen to be solved
# together; they are the three components of one displacement. If you
# never need them apart, the multi-field machinery is bookkeeping you are paying
# for and not using: two spaces to build, blocked trial and test functions to
# unpack in every term, a solution that comes back in pieces.
#
# `CartProdRefFE(r₁, r₂, r₃)` builds the product element instead. The factors
# must share a polytope and a value type — that is what makes the result one
# `MultiValue` rather than a multi-field space — but they need share nothing
# else, neither degree nor conformity. Here all three factors are scalar
# elements on a triangle, so the product is `VectorValue{3}`-valued, with
# factor $c$ occupying component $c$ of the value and block $c$ of the degrees
# of freedom.

using Gridap
using Gridap.ReferenceFEs
using Gridap.TensorValues

# ## Discrete model and space
#
# Argyris is defined on triangles, so we simplexify a Cartesian mesh.
#
# The membrane components are quadratic Lagrange and the bending component is
# quintic Argyris. Their conformities differ — `H1` against `H2` — so the
# product reports a `CartProdConformity`, which applies each factor's own
# ownership rules to its own block. That is the whole mechanism: the first two
# components glue like Lagrange, the third like Argyris.

n = 8
model = simplexify(CartesianDiscreteModel((0, 1, 0, 1), (n, n)))

membrane_reffe = ReferenceFE(TRI, lagrangian, Float64, 2)
bending_reffe = ReferenceFE(TRI, argyris, Float64, 5)
reffe = CartProdRefFE(membrane_reffe, membrane_reffe, bending_reffe)

println("dofs per cell: ", num_dofs(reffe),
        " = ", num_dofs(membrane_reffe), " + ", num_dofs(membrane_reffe),
        " + ", num_dofs(bending_reffe))
println("conformity: ", Conformity(reffe))

# The plate is clamped all round: $u = 0$ and $w = \partial_n w = 0$ on
# $\partial\Omega$. Setting every boundary degree of freedom to zero also pins
# the vertex Hessians of $w$, which is slightly more than a clamp requires — the
# Argyris tutorial avoids this by prescribing a manufactured solution on the
# boundary. Here we keep the homogeneous data, and note that every comparison
# below is between discretisations that share it.

V = FESpace(model, reffe; dirichlet_tags="boundary")
U = TrialFESpace(V)

Ω = Triangulation(model)
dΩ = Measure(Ω, 12)

# ## Kinematics
#
# This is where one field pays. The unknown `U` is `VectorValue{3}`-valued on a
# two-dimensional domain, so `∇(U)` is $2\times3$ with
# $(\nabla U)_{ij} = \partial_i U_j$, and `∇∇(U)` is $2\times2\times3$. Both the
# membrane strain and the curvature are slices of those two objects, obtained by
# contracting the *component* index:
#
# ```math
# \varepsilon(u) = \operatorname{sym}\big( \nabla U \cdot P \big), \qquad
# \kappa(w) = \nabla\nabla U \cdot e_3 ,
# ```
#
# where $P$ keeps the first two components and $e_3$ the third. Contraction with
# `⋅` acts on the last index of its left argument, which is the component index
# in both cases, so the derivative indices are untouched.

P = TensorValue{3,2}(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
e3 = VectorValue(0.0, 0.0, 1.0)

displacement(U) = U ⋅ P
deflection(U) = U ⋅ e3
strain(U) = symmetric_part( ∇(U) ⋅ P )
curvature(U) = ∇∇(U) ⋅ e3

# ## Weak form and solve
#
# The load is a uniform body force with an in-plane and a transverse part. It
# enters as `∫( f ⋅ v )`, a single contraction against the whole test function —
# there is no unpacking to do.

f = VectorValue(1.0, 1.0, 1.0)

function solve_laminate(b)
  a(u, v) = ∫( strain(u) ⊙ strain(v)
               + b * ( strain(u) ⊙ curvature(v) + curvature(u) ⊙ strain(v) )
               + curvature(u) ⊙ curvature(v) )dΩ
  l(v) = ∫( f ⋅ v )dΩ
  solve(AffineFEOperator(a, l, U, V))
end

Uh = solve_laminate(0.5)

println("compliance: ", sum( ∫( f ⋅ Uh )dΩ ))

# ## At $b = 0$ the product is exactly the two spaces
#
# With the coupling switched off the energy splits, and the product space must
# reproduce, component by component, what the two spaces give on their own. We
# solve the same two problems in a plain vector Lagrange space and a plain
# Argyris space and compare. The degrees of freedom add up exactly, and so do
# the solutions.

U0 = solve_laminate(0.0)

Vw = FESpace(model, ReferenceFE(argyris, Float64, 5); dirichlet_tags="boundary")
aw(w, v) = ∫( ∇∇(w) ⊙ ∇∇(v) )dΩ
lw(v) = ∫( 1.0 * v )dΩ
w_solo = solve(AffineFEOperator(aw, lw, TrialFESpace(Vw), Vw))

Vu = FESpace(model, ReferenceFE(lagrangian, VectorValue{2,Float64}, 2);
             dirichlet_tags="boundary")
au(u, v) = ∫( ε(u) ⊙ ε(v) )dΩ
lu(v) = ∫( VectorValue(1.0, 1.0) ⋅ v )dΩ
u_solo = solve(AffineFEOperator(au, lu, TrialFESpace(Vu), Vu))

println("free dofs: ", num_free_dofs(V),
        " = ", num_free_dofs(Vu), " + ", num_free_dofs(Vw))

ew = deflection(U0) - w_solo
eu = displacement(U0) - u_solo
println("deflection difference:   ", sqrt(sum( ∫( ew * ew )dΩ )))
println("displacement difference: ", sqrt(sum( ∫( eu ⋅ eu )dΩ )))

# ## The coupling
#
# Now load the plate transversely only, with $f = (0,0,1)$, and vary $b$. A
# symmetric laminate does not stretch under a transverse load, so the in-plane
# displacement is *exactly* zero at $b = 0$ — the two blocks of the stiffness
# matrix do not talk to each other, and the in-plane load is what was removed.
# As soon as $b \neq 0$ it is not, and the plate also deflects more: the
# stretching it is now free to do lowers the energy at fixed load.

f_transverse = VectorValue(0.0, 0.0, 1.0)

for b in (0.0, 0.25, 0.5, 0.75)
  a(u, v) = ∫( strain(u) ⊙ strain(v)
               + b * ( strain(u) ⊙ curvature(v) + curvature(u) ⊙ strain(v) )
               + curvature(u) ⊙ curvature(v) )dΩ
  l(v) = ∫( f_transverse ⋅ v )dΩ
  Ub = solve(AffineFEOperator(a, l, U, V))
  nu = sqrt(sum( ∫( displacement(Ub) ⋅ displacement(Ub) )dΩ ))
  nw = sqrt(sum( ∫( deflection(Ub) * deflection(Ub) )dΩ ))
  println("b = ", b, "   |u| = ", nu, "   |w| = ", nw,
          "   compliance = ", sum( ∫( f_transverse ⋅ Ub )dΩ ))
end

# ## One field, two conformities
#
# The claim to check is that the components really do glue differently. We
# interpolate a smooth function into the *unconstrained* product space and look
# at the jumps across the skeleton, component by component.
#
# The whole field is $C^0$, as both factors are. But the gradient jumps in the
# membrane components, which are only Lagrange, and vanishes in the bending
# component, which is Argyris. One element, two continuities.

Vfree = FESpace(model, reffe)
g(x) = VectorValue(sin(2*x[1]) * x[2], cos(3*x[2]), sin(2*x[1]) * cos(3*x[2]))
gh = interpolate(g, Vfree)

Λ = SkeletonTriangulation(model)
dΛ = Measure(Λ, 12)

jump_u = jump(∇(gh)) ⋅ P
jump_w = jump(∇(gh)) ⋅ e3

println("value jump, whole field:      ",
        sqrt(sum( ∫( jump(gh) ⋅ jump(gh) )dΛ )))
println("gradient jump, membrane (P2): ",
        sqrt(sum( ∫( jump_u ⊙ jump_u )dΛ )))
println("gradient jump, bending (Arg): ",
        sqrt(sum( ∫( jump_w ⋅ jump_w )dΛ )))

# ## Visualisation
#
# The solution is a single `FEFunction`, so it is written out as one field.

mkpath("output_path")
writevtk(Ω, "output_path/laminate_plate",
         cellfields=["U" => Uh,
                     "u" => displacement(Uh),
                     "w" => deflection(Uh)])
