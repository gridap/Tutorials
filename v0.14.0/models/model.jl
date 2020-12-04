
using Gridap
using Gridap.Io
using GridapGmsh

model = GmshDiscreteModel("model.msh")

writevtk(model,"model")

fn = "model.json"
to_json_file(model,fn)

