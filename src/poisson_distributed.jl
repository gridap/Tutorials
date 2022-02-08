# ## Introduction and caveat

# In this tutorial we will learn how to use `GridapDistributed.jl` and its satellite packages, `GridapP4est.jl`, `GridapGmsh.jl`, and `GridapPETSc.jl`, in order to solve a Poisson PDE problem  on the unit square using grad-conforming Lagrangian Finite Elements for numerical discretization.

# We will first solve the problem using solely the built-in tools in `GridapDistributed.jl`. While this is very useful for testing and debugging purposes, `GridapDistributed.jl` is **not** a library of parallel solvers. Indeed, the built-in linear solver kernel within `GridapDistributed.jl`, defined with the backslash operator `\`, is just a sparse LU solver applied to the global system gathered on a master task (thus not scalable). To address this, we will then illustrate which changes are required in the program to switch the built-in solver in `GridapDistributed.jl` to `GridapPETSc.jl`. This package provides the full set of scalable linear and nonlinear solvers in the [PETSc](https://petsc.org/release/) numerical package.

# On the other hand, in real-world applications, one typically needs to solve PDEs on more complex domains than simple boxes. To this end, we can leverage either `GridapGmsh.jl`, in order to partition and distribute automatically unstructured meshes read from disk in gmsh format, or `GridapP4est.jl`, which allows one to mesh in a very scalable way computational domains which can be decomposed as forests of octrees. The last part of the tutorial will present the necessary changes in the program in order to use these packages.

# **IMPORTANT NOTE**: the parallel codes in this tutorial depend on the Message Passing Interface (MPI). Thus, they cannot be easily executed interactively, e.g., in a Jupyter notebook. Instead, one has to run them from a terminal using the [`mpiexecjl`](https://juliaparallel.github.io/MPI.jl/stable/configuration/#Julia-wrapper-for-mpiexec) script as provided by [MPI.jl](https://github.com/JuliaParallel/MPI.jl), e.g., with the command `mpiexecjl --project=. -n 4 julia src/poisson_distributed.jl` run from the root directory of the Tutorials git repository.

# ## First example: `GridapDistributed.jl` built-in tools

using Gridap
using GridapDistributed
using PartitionedArrays

# The first step in any `GridapDistributed.jl` program is to define a function (named `main` below) to be executed on each part on which the domain is distributed. This function receives a single argument (named `parts` below). The body of this function is equivalent to a sequential `Gridap` script, except for the `CartesianDiscreteModel` call, which in `GridapDistributed` also requires the `parts` argument passed to the `main` function. The domain is discretized using the parallel Cartesian-like mesh generator built-in in `GridapDistributed`.

function main(parts)
  domain = (0,1,0,1)
  mesh_partition = (4,4)
  model = CartesianDiscreteModel(parts,domain,mesh_partition)
  order = 2
  u((x,y)) = (x+y)^order
  f(x) = -Δ(u,x)
  reffe = ReferenceFE(lagrangian,Float64,order)
  V = TestFESpace(model,reffe,dirichlet_tags="boundary")
  U = TrialFESpace(u,V)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,2*order)
  a(u,v) = ∫( ∇(v)⋅∇(u) )dΩ
  l(v) = ∫( v*f )dΩ
  op = AffineFEOperator(a,l,U,V)
  uh = solve(op)
  writevtk(Ω,"results_example1",cellfields=["uh"=>uh,"grad_uh"=>∇(uh)])
end

# Once the `main` function has been defined, we have to trigger its execution on the different parts. To this end, once calls the `prun` function of [`PartitionedArrays.jl`](https://github.com/fverdugo/PartitionedArrays.jl) right at the beginning of the program.

partition = (2,2)
prun(main, mpi,partition)

# With this function, the programer sets up the `PartitionedArrays.jl` communication backend (i.e., MPI in the example), specifies the number of parts and their layout (i.e., 2x2 Cartesian-like mesh partition in the example), and provides the `main` function to be run on each part.

# ## Second example: `GridapDistributed.jl` + `GridapPETSc.jl` for the linear solver

using GridapPETSc

# TBD: Descriptive text goes here ... Focus on the differences compared to example 1

function main(parts)
  options = "-ksp_type cg -pc_type gamg -ksp_monitor"
  GridapPETSc.with(args=split(options)) do
    domain = (0,1,0,1)
    mesh_partition = (4,4)
    model = CartesianDiscreteModel(parts,domain,mesh_partition)
    order = 2
    u((x,y)) = (x+y)^order
    f(x) = -Δ(u,x)
    reffe = ReferenceFE(lagrangian,Float64,order)
    V = TestFESpace(model,reffe,dirichlet_tags="boundary")
    U = TrialFESpace(u,V)
    Ω = Triangulation(model)
    dΩ = Measure(Ω,2*order)
    a(u,v) = ∫( ∇(v)⋅∇(u) )dΩ
    l(v) = ∫( v*f )dΩ
    op = AffineFEOperator(a,l,U,V)
    solver = PETScLinearSolver()
    uh = solve(solver,op)
    writevtk(Ω,"results_example2",cellfields=["uh"=>uh,"grad_uh"=>∇(uh)])
  end
end

partition = (2,2)
prun(main, mpi, partition)

# ## Third example: second example + `GridapP4est.jl` for mesh generation

# TBD: Descriptive text goes here ... Focus on the differences compared to example 2. Introduce the concept of coarse_discrete_model and uniform refinement, etc. We do not longer use a Cartesian partition but just the number of parts when calling prun.

using GridapP4est

function main(parts)
  options = "-ksp_type cg -pc_type gamg -ksp_monitor"
  GridapPETSc.with(args=split(options)) do
    domain = (0,1,0,1)
    coarse_mesh_partition = (1,1)
    num_uniform_refinements=2
    coarse_discrete_model=CartesianDiscreteModel(domain,coarse_mesh_partition)
    model=UniformlyRefinedForestOfOctreesDiscreteModel(parts,
                                                       coarse_discrete_model,
                                                       num_uniform_refinements)
    order = 2
    u((x,y)) = (x+y)^order
    f(x) = -Δ(u,x)
    reffe = ReferenceFE(lagrangian,Float64,order)
    V = TestFESpace(model,reffe,dirichlet_tags="boundary")
    U = TrialFESpace(u,V)
    Ω = Triangulation(model)
    dΩ = Measure(Ω,2*order)
    a(u,v) = ∫( ∇(v)⋅∇(u) )dΩ
    l(v) = ∫( v*f )dΩ
    op = AffineFEOperator(a,l,U,V)
    solver = PETScLinearSolver()
    uh = solve(solver,op)
    writevtk(Ω,"results_example3",cellfields=["uh"=>uh,"grad_uh"=>∇(uh)])
  end
end

nparts = 4
prun(main, mpi, nparts)

# ## Fourth example: second example + `GridapGmsh.jl` for mesh generation

using GridapGmsh

# TBD
