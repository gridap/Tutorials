module Tutorials

deps_jl = joinpath(@__DIR__, "..", "deps", "deps.jl")

if !isfile(deps_jl)
  s = """
  Package Tutorials not installed properly.
  Run Pkg.build(\"Tutorials\"), restart Julia and try again
  """
  error(s)
end

include(deps_jl)

end # module
