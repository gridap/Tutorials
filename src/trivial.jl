# In this tutorial we will learn
#
#    - How to solve the trivial equation `u = f`
#    - How to visualize the solution in pure julia
#
# We want to solve `u = sin`. This equation is trivial, but it showcases how
# the finite element machinery works. The first step is to rephrase it as
# a variational problem:
# ```math
# \int u \cdot v dx = \int sin \cdot v dx
# ```
# for all test functions `v`.
using Gridap
model = CartesianDiscreteModel((0, 2π), 10) # partition the interval (0, 2π) into 10 cells
f(pt) = sin(pt[1])
V0 = TestFESpace(
  reffe=:Lagrangian, order=1, valuetype=Float64,
  conformity=:H1, model=model)
U = TrialFESpace(V0)
trian = Triangulation(model)
degree = 2
quad = CellQuadrature(trian, degree)
A(u, v) = u ⊙ v
b(v) = v*f
t_Ω = AffineFETerm(A,b,trian,quad)
op = AffineFEOperator(U,V0,t_Ω)
u = solve(op)

# Now that we have a solution, we want to know if it is any good.
# So lets visualize it.
using Plots
xs = map(get_cell_coordinates(trian)) do cell
    left, right = cell
    1/2*(left[1] + right[1])
end # physical coordinges of the cell centers
q = fill([VectorValue((1/2,))], length(xs)) # reference coordinates of each cell center.
ys = only.(evaluate(u,q)) # solution values at the cell centers
plot(xs, ys, label="solution")
plot!(sin, 0:0.01:2pi, label="truth")
