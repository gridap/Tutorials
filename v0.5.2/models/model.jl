
using Gridap
using GridapGmsh
using JSON

model = GmshDiscreteModel("model.msh")

writevtk(model,"model")

fn = "model.json"
open(fn,"w") do f
  JSON.print(f,model)
end
