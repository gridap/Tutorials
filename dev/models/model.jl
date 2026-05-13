
using Gridap
using Gridap.Io
using GridapGmsh

meshdir(args...) = joinpath(@__DIR__, args...)
model = GmshDiscreteModel(meshdir("model.msh"))

writevtk(model,meshdir("model"))
 
to_json_file(model,meshdir("model.json"))
