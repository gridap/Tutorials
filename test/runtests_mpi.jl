module TutorialsMPITests

using MPI
using Test

#Sysimage
sysimage=nothing
if length(ARGS)==1
   @assert isfile(ARGS[1]) "$(ARGS[1]) must be a valid Julia sysimage file"
   sysimage=ARGS[1]
end

mpidir  = @__DIR__
repodir = joinpath(mpidir,"..")
srcdir = joinpath(mpidir,"../src/")
function run_driver(procs,file,sysimage)
  mpiexec() do cmd
    if sysimage!=nothing
       extra_args="-J$(sysimage)"
       run(`$cmd -n $procs $(Base.julia_cmd()) $(extra_args) --project=$repodir $(joinpath(mpidir,file))`)
    else
       run(`$cmd -n $procs $(Base.julia_cmd()) --project=$repodir $(joinpath(mpidir,file))`)
    end
    @test true
  end
end

run_driver(4,joinpath(srcdir,"poisson_distributed.jl"),sysimage)


end # module
