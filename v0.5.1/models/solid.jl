
using Gridap
using GridapGmsh
using JSON

model = GmshDiscreteModel("solid.msh")

writevtk(model,"solid")

fn = "solid.json"
open(fn,"w") do f
  JSON.print(f,model)
end
