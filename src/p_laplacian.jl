
using Gridap

const p = 3

norm(u) = sqrt(inner(u,u)) #TODO dot, norm

@law j(x,∇u) = norm(∇u)^(p-2) * ∇u

@law dj(x,∇du,∇u) = (p-2)*norm(∇u)^(p-4)*inner(∇u,∇du)*∇u + norm(∇u)^(p-2) * ∇du

f(x) = 0.0

res(u,v) = inner( ∇(v), j(∇(u)) ) - inner(v,f)

jac(u,v,du) = inner(  ∇(v) , dj(∇(du),∇(u)) )

h(x) = 0.0

neum(v) = inner(v,h)

model = CartesianDiscreteModel(domain=(0,1,0,1,0,1),partition=(10,10,10))

labels = FaceLabels(model)

