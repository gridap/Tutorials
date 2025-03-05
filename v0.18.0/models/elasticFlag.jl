using Gridap
using Gridap.Io
using GridapGmsh

model = GmshDiscreteModel("elasticFlag.msh")

writevtk(model,"elasticFlag")

fn = "elasticFlag.json"
to_json_file(model,fn)
