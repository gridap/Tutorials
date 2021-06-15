using Gridap
using Gridap.Io
using GridapGmsh

model = GmshDiscreteModel("cylinder_NSI.msh")

writevtk(model,"cilinder_NSI")

fn = "cylinder_NSI.json"
to_json_file(model,fn)
