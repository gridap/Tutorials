# ## Introduction and caveat

# In this tutorial we will learn how to use `GridapDistributed.jl` and its satellite packages, `GridapP4est.jl`, `GridapGmsh.jl`, and `GridapPETSc.jl`, in order to solve a Poisson PDE problem  on the unit square using grad-conforming Lagrangian Finite Elements for numerical discretization.

# We will first solve the problem using solely the built-in tools in `GridapDistributed.jl`. While this is very useful for testing and debugging purposes, `GridapDistributed.jl` is **not** a library of parallel solvers. Indeed, the built-in linear solver kernel within `GridapDistributed.jl`, defined with the backslash operator `\`, is just a sparse LU solver applied to the global system gathered on a master task (thus not scalable). To address this, we will then illustrate which changes are required in the program to replace the built-in solver in `GridapDistributed.jl` by `GridapPETSc.jl`. This latter package provides the full set of scalable linear and nonlinear solvers in the [PETSc](https://petsc.org/release/) numerical package.

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
  writevtk(Ω,"results_ex1",cellfields=["uh"=>uh,"grad_uh"=>∇(uh)])
end

# Once the `main` function has been defined, we have to trigger its execution on the different parts. To this end, one calls the `prun` function of [`PartitionedArrays.jl`](https://github.com/fverdugo/PartitionedArrays.jl) right at the beginning of the program.

partition = (2,2)
prun(main, mpi,partition)

# With this function, the programmer sets up the `PartitionedArrays.jl` communication backend (i.e., MPI in the example), specifies the number of parts and their layout (i.e., 2x2 Cartesian-like mesh partition in the example), and provides the `main` function to be run on each part.

# Although not illustrated in this tutorial, we note that one may also use the `sequential` `PartitionedArrays.jl` backend, instead of `mpi`. With this backend, the code executes serially on a single process (and there is thus no need to use `mpiexecjl` to launch the program), although  the data structures are still partitioned into parts. This is very useful, among others, for interactive execution of the code, and debugging, before moving to MPI parallelism.

# ## Second example: `GridapDistributed.jl` + `GridapPETSc.jl` for the linear solver

using GridapPETSc

# The example code that leverages `GridapPETSc.jl` is almost identical as the one above (see below). The main difference is that now we are wrapping most of the code of the `main` function within a do-block syntax function call to the `GridapPETSc.with(args=split(options))` function. The `with` function receives as a first argument a function with no arguments with the instructions to be executed on each MPI task/subdomain (that we pass to it as an anonymous function with no arguments), along with the `options` to be passed to the PETSc linear solver. For a detailed explanation of possible options we refer to the PETSc library documentation. Note that the call to `PETScLinearSolver()` initializes the PETSc solver with these `options` (even though `options` is not actually passed to the linear solver constructor). Besides, we have to pass the created linear solver object `solver` to the `solve` function to override the default linear solver (i.e., a call to the backslash `\` Julia operator).

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
    writevtk(Ω,"results_ex2",cellfields=["uh"=>uh,"grad_uh"=>∇(uh)])
  end
end

partition = (2,2)
prun(main, mpi, partition)

# ## Third example: second example + `GridapP4est.jl` for mesh generation

# Using `GridapP4est.jl` for mesh generation is very simple, and only involves minor modifications compared to the previous example. First, one has to generate a coarse mesh of the domain. As the domain is a just a simple box in the example, it suffices to use a coarse mesh with a single quadrilateral fitted to the box in order to capture the geometry of the domain with no geometrical error (see how the `coarse_discrete_model` object is generated). In more complex scenarios, one can read an unstructured coarse mesh from disk, generated, e.g., with an unstructured brick mesh generator. Second, when building the fine mesh of the domain (see `UniformlyRefinedForestOfOctreesDiscreteModel` call), one has to specify the number of uniform refinements to be performed on the coarse mesh in order to generate the fine mesh. Finally, when calling `prun`, we do not longer specify a Cartesian partition but just the number of parts.

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
    writevtk(Ω,"results_ex3",cellfields=["uh"=>uh,"grad_uh"=>∇(uh)])
  end
end

nparts = 4
prun(main, mpi, nparts)

# ## Fourth example: second example + `GridapGmsh.jl` for mesh generation

# The only modification with respect to the second example driver above is that now the mesh is read from disk and partitioned/distributed automatically by `GridapGmsh` inside the call to the `GmshDiscreteModel` constructor.

using GridapGmsh
function main(parts)
  options = "-ksp_type cg -pc_type gamg -ksp_monitor"
  GridapPETSc.with(args=split(options)) do
    model = GmshDiscreteModel(parts,"../models/demo.msh")
    order = 2
    u((x,y)) = (x+y)^order
    f(x) = -Δ(u,x)
    reffe = ReferenceFE(lagrangian,Float64,order)
    V = TestFESpace(model,reffe,dirichlet_tags=["boundary1","boundary2"])
    U = TrialFESpace(u,V)
    Ω = Triangulation(model)
    dΩ = Measure(Ω,2*order)
    a(u,v) = ∫( ∇(v)⋅∇(u) )dΩ
    l(v) = ∫( v*f )dΩ
    op = AffineFEOperator(a,l,U,V)
    solver = PETScLinearSolver()
    uh = solve(solver,op)
    writevtk(Ω,"results_ex4",cellfields=["uh"=>uh,"grad_uh"=>∇(uh)])
  end
end

nparts = 4
prun(main, mpi, nparts)
