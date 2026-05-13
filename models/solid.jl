
using Gridap
using Gridap.Io
using GridapGmsh

meshdir(args...) = joinpath(@__DIR__, args...)
model = GmshDiscreteModel(meshdir("solid.msh"))

writevtk(model,meshdir("solid"))

to_json_file(model,meshdir("solid.json"))
