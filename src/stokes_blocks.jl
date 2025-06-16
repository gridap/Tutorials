# # Incompressible Stokes equations in a 2D/3D cavity
# 
# This example solves the incompressible Stokes equations, given by 
# 
# ```math
# \left\lbrace
# \begin{aligned}
# -\Delta u - \nabla p &= f \quad \text{in} \quad \Omega, \\
# \nabla \cdot u &= 0 \quad \text{in} \quad \Omega, \\
# u &= \hat{x} \quad \text{in} \quad \Gamma_\text{top} \subset \partial \Omega, \\
# u &= 0 \quad \text{in} \quad \partial \Omega \backslash \Gamma_\text{top} \\
# \end{aligned}
# \right.
# ```
# 
# where $\Omega = [0,1]^d$. 
# 
# We use a mixed finite-element scheme, with $Q_k \times P_{k-1}^{-}$ elements for the velocity-pressure pair. 
# 
# To solve the linear system, we use a FGMRES solver preconditioned by a block-diagonal or 
# block-triangular Shur-complement-based preconditioner. 
# 
# 
# ## Block structure
#
# The discretized system has a natural 2×2 block structure:
#
# ```math
# \begin{bmatrix} 
# A & B^T \\ 
# B & 0
# \end{bmatrix}
# \begin{bmatrix}
# u \\ 
# p
# \end{bmatrix} = 
# \begin{bmatrix}
# f \\
# 0
# \end{bmatrix}
# ```
#
# where:
# - $A$: Vector Laplacian (velocity block)
# - $B$: Divergence operator 
# - $B^T$: Gradient operator
#
# ## Solution strategy
#
# We use a FGMRES solver preconditioned by an upper block-triangular preconditioner:
#
# ```math
# P = \begin{bmatrix}
# A & B^T \\
# 0 & -\hat{S}
# \end{bmatrix}
# ```
#
# where $\hat{S}$ is an approximation of the Schur complement $S = BA^{-1}B^T$, which we 
# will approximate using a pressure mass matrix.

using LinearAlgebra
using BlockArrays

using Gridap
using Gridap.MultiField

using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.BlockSolvers: LinearSystemBlock, BiformBlock
using GridapSolvers.BlockSolvers: BlockDiagonalSolver, BlockTriangularSolver

# ## Geometry and FESpaces
#
# See the basic Stokes tutorial for a detailed explanation.
#

nc = (8,8)
Dc = length(nc)
domain = (Dc == 2) ? (0,1,0,1) : (0,1,0,1,0,1)

model = CartesianDiscreteModel(domain,nc)
labels = get_face_labeling(model)
if Dc == 2
  add_tag_from_tags!(labels,"top",[6])
  add_tag_from_tags!(labels,"walls",[1,2,3,4,5,7,8])
else
  add_tag_from_tags!(labels,"top",[22])
  add_tag_from_tags!(labels,"walls",[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26])
end

order = 2
qdegree = 2*(order+1)
reffe_u = ReferenceFE(lagrangian,VectorValue{Dc,Float64},order)
reffe_p = ReferenceFE(lagrangian,Float64,order-1;space=:P)

u_walls = (Dc==2) ? VectorValue(0.0,0.0) : VectorValue(0.0,0.0,0.0)
u_top = (Dc==2) ? VectorValue(1.0,0.0) : VectorValue(1.0,0.0,0.0)

V = TestFESpace(model,reffe_u,dirichlet_tags=["walls","top"]);
U = TrialFESpace(V,[u_walls,u_top]);
Q = TestFESpace(model,reffe_p;conformity=:L2,constraint=:zeromean) 

# ## Block multi-field spaces
#
# Our first difference will come from how we define our multi-field spaces: 
# Because we want to be able to use the block-structure of the linear system, 
# we have to assemble our problem by blocks. The block structure of the resulting 
# linear system is determined by the `BlockMultiFieldStyle` we use to define the
# multi-field spaces.
#
# A `BlockMultiFieldStyle` takes three arguments:
#   - `N`: The number of blocks we want
#   - `S`: An N-tuple with the number of fields in each block
#   - `P`: A permutation of the fields in the multi-field space, which determines
#          how fields are grouped into blocks.
#
# By default, we create as many blocks as there are fields in the multi-field space,
# and each block contains a single field with no permutation.

mfs = BlockMultiFieldStyle(2,(1,1),(1,2))
X = MultiFieldFESpace([U,Q];style=mfs)
Y = MultiFieldFESpace([V,Q];style=mfs)

# ## Weak form and integration

Ω = Triangulation(model)
dΩ = Measure(Ω,qdegree)

α = 1.e1
f = (Dc==2) ? VectorValue(1.0,1.0) : VectorValue(1.0,1.0,1.0)
a((u,p),(v,q)) = ∫(∇(v)⊙∇(u))dΩ - ∫(divergence(v)*p)dΩ - ∫(divergence(u)*q)dΩ
l((v,q)) = ∫(v⋅f)dΩ

op = AffineFEOperator(a,l,X,Y)

# ### Block structure of the linear system
#
# As per usual, we can extract the matrix and vector of the linear system from the operator.
# Notice now that unlike in previous examples, the matrix is a `BlockMatrix` type, from 
# the `BlockArrays.jl` package, which allows us to work with block-structured matrices.

A = get_matrix(op)
b = get_vector(op)

# ## Block solvers
#
# We will now setup two types of block preconditioners for the Stokes system. In both cases, 
# we will use the preconditioners to solve the linear system using a Flexible GMRES solver.
# The idea behind these preconditioners is the well-known property that the Schur complement
# of the velocity block can be well approximated by a scaled pressure mass matrix. Moreover, 
# in our pressure discretization is discontinuous which means that the pressure mass matrix 
# is block-diagonal and easily invertible.
# In this example, we will use an exact LU solver for the velocity block and a CG solver 
# with Jacobi preconditioner for the pressure block.

# ### Block diagonal preconditioner
#
# The simplest block preconditioner is the block-diagonal preconditioner.
# The only ingredients required are 
#  - the sub-solvers for each diagonal block and 
#  - the diagonal blocks we want to use.
#
# The sub-solvers are defined as follows:

u_solver = LUSolver()
p_solver = CGSolver(JacobiLinearSolver();maxiter=20,atol=1e-14,rtol=1.e-6)

# The block structure is defined using the block API. We provide different types of blocks, 
# that might have different uses depending on the problem at hand. We will here use two of 
# the most common block types:
#  - The `LinearSystemBlock` defines a block that is taken directly (and aliased) form
#    the linear system matrix `A`. We will use this for the velocity block. 
#  - For the pressure block, however, the pressure mass matrix is not directly available
#    in the system matrix. Instead, we will have to integrate it using the Gridap API, as usual. 
#    These abstract concept is implemented in the `BiformBlock` type, which allows the 
#    user to define a block from a bilinear form.
# All in all, we define the block structure as follows:

u_block = LinearSystemBlock()
p_block = BiformBlock((p,q) -> ∫(-(1.0/α)*p*q)dΩ,Q,Q)

# With these ingredients, we can now define the block diagonal preconditioner as follows:

PD = BlockDiagonalSolver([u_block,p_block],[u_solver,p_solver])
solver_PD = FGMRESSolver(20,PD;atol=1e-10,rtol=1.e-12,verbose=true)

uh, ph = solve(solver_PD, op)

# ### Block upper-triangular preconditioner
#
# A slighly more elaborate preconditioner (but also more robust) is the 
# block upper-triangular preconditioner. The ingredients are similar: 
#  - the sub-solvers for each diagonal block
#  - the blocks that define the block structure, now including the off-diagonal blocks
#  - the coefficients for the off-diagonal blocks, where zero coefficients
#    indicate that the block is not used.
#
# We will also represent the off-diagonal blocks using the `LinearSystemBlock` type.

sblocks = [     u_block        LinearSystemBlock();
           LinearSystemBlock()      p_block       ]
coeffs = [1.0 1.0;
          0.0 1.0]
PU = BlockTriangularSolver(sblocks,[u_solver,p_solver],coeffs,:upper)
solver_PU = FGMRESSolver(20,PU;atol=1e-10,rtol=1.e-12,verbose=true)

uh, ph = solve(solver_PU, op)

# As you can see, the block upper-triangular preconditioner is quite better 
# than the block diagonal one.

# ## Going further
#
# If you want to see more examples of how to use the block solvers,
# you can check the documentation in [GridapSolvers.jl](https://gridap.github.io/GridapSolvers.jl/stable/),
# as well as it's `test/Applications` folder.
# 
# There you will find more complicated examples, such as using a GMG solver to 
# solve the velocity block. 
