# # Tutorial 3: Linear elasticity
#
#md # [![](https://mybinder.org/badge_logo.svg)](@__BINDER_ROOT_URL__/notebooks/t003_elasticity.ipynb)
#md # [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/notebooks/t003_elasticity.ipynb)
# 
# ## Learning outcomes
#
# - How to approximate vector-valued problems
# - How to solve problems with complex constitutive laws
# - How to impose Dirichlet boundary conditions only in selected components
# - How to impose Dirichlet boundary conditions described by more than one function
#
# ## Problem statement
#
# ![](../models/solid.png)

# Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.


# ## Constitutive law
#

using Gridap

const E = 70.0e9
const ν = 0.33

const λ = (E*ν)/((1+ν)*(1-2*ν))
const μ = E/(2*(1+ν))

@law σ(x,ε) = λ*tr(ε)*one(ε) + 2*μ*ε

# ## Vector-valued FE space

model = DiscreteModelFromFile("../models/solid.json");

const T = VectorValue{3,Float64}
diritags = ["surface_1","surface_2"]
dirimasks = [(true,false,false), (true,true,true)]

order = 1
V = CLagrangianFESpace(T,model,order,diritags,dirimasks)

g1(x) = VectorValue(0.005,0.0,0.0)
g2(x) = zero(T)
U = TrialFESpace(V,[g1,g2])
V0 = TestFESpace(V)

trian = Triangulation(model)
quad = CellQuadrature(trian,order=2)

a(v,u) = inner( ε(v), σ(ε(u)) )
t_Ω = LinearFETerm(a,trian,quad)

op = LinearFEOperator(V0,U,t_Ω)

uh = solve(op)

writevtk(trian,"results",cellfields=["uh"=>uh,"epsi"=>ε(uh),"sigma"=>σ(ε(uh))])

# Deformation magnified 40 times
# ![](../assets/t003_elasticity/disp_ux_40.png)


# ## Multi-material solids
#
# ![](../models/solid-mat.png)
#

function lame_parameters(E,ν)
  λ = (E*ν)/((1+ν)*(1-2*ν))
  μ = E/(2*(1+ν))
  (λ, μ)
end

const E_alu = 70.0e9
const ν_alu = 0.33
const (λ_alu,μ_alu) = lame_parameters(E_alu,ν_alu)

const E_steel = 200.0e9
const ν_steel = 0.33
const (λ_steel,μ_steel) = lame_parameters(E_steel,ν_steel)


# **Most of the following API is not yet implemented!**

@law function σ_bimat(x,ε,mask)
  if mask
    return λ_alu*tr(ε)*one(ε) + 2*μ_alu*ε
  else
    return λ_steel*tr(ε)*one(ε) + 2*μ_steel*ε
  end
end

mask = is_cell_on_tag(model,"material_1")

a(v,u) = inner( ε(v), σ_bimat(ε(u),mask) )
t_Ω = LinearFETerm(a,trian,quad)

op = LinearFEOperator(V0,U,t_Ω)

uh = solve(op)

writevtk(trian,"results",cellfields=
  ["uh"=>uh,"epsi"=>ε(uh),"sigma"=>σ_bimat(ε(uh),mask)])


#
#
#
#
#
#  Tutorial done!
