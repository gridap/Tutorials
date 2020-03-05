module Tutorials

deps_jl = joinpath(@__DIR__, "..", "deps", "deps.jl")

if !isfile(deps_jl)
  error("Package Tutorials not installed properly.")
end

include(deps_jl)

end # module
