using Gridap
using Gridap.Io
using GridapGmsh

meshdir(args...) = joinpath(@__DIR__, args...)
model = GmshDiscreteModel(meshdir("elasticFlag.msh"))

writevtk(model,meshdir("elasticFlag"))

to_json_file(model,meshdir("elasticFlag.json"))
