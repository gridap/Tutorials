# In this tutorial, we will learn:
# 
#   * How to formulate the weak form for time-harmonic electromagnetic problem
#   * How to implement perfectly matched layers (PML)
#   * How does periodic boundary condition work in Gridap
# 
# # Problem statement
# 
# The problem that we are trying to solve is the plane wave (TM-polarized $H_{inc}$) scattering of a cylinder dielectric (of radius $R$ and permittivity $\varepsilon$), as illustrated below. The computational cell is of height $H$ and length $L$, and we designate a perfectly matched layer (PML) thickness of $d_{pml}$.
# 
# ![](../assets/emscatter/Illustration.png)
# 
# The governing equation of this scaler electromagnetic problem in 2D is the Helmholtz equation: 
#
# ```math
# \left[-\nabla\cdot\frac{1}{\varepsilon(x)}\cdot\nabla -k^2\mu(x)\right] H = f(x),
# ```
#
# where $k=\omega/c$ is the wave number in free space and $f(x)$ is the source term
# 
# The boundary conditions are zero Dirichlet boundary on the top and bottom side $\Gamma_D$ and periodic boundary condition on the left ($\Gamma_L$) and right side ($\Gamma_R$). The PML is implemented by a coordinate streching: 
#
# ```math
# \frac{\partial}{\partial x}\rightarrow \frac{1}{1+\mathrm{i}\sigma(u)/\omega}\frac{\partial}{\partial x},
# ```
#
# ```math
# \frac{\partial}{\partial y}\rightarrow \frac{1}{1+\mathrm{i}\sigma(u)/\omega}\frac{\partial}{\partial y},
# ```
#
# where $u$ is the depth into the PML, $\sigma$ is a profile function (here we chose $\sigma(u)=\sigma_0u^2$) and different derivative denotes different absorption direction. 
# 
# Consider $\mu(x)=1$ and denote $\Lambda=\frac{1}{1+\mathrm{i}\sigma(u)/\omega}$, we can formulate the problem as 
#
# ```math
# \left\{ \begin{aligned} 
# \left[-\Lambda\nabla\cdot\frac{1}{\varepsilon(x)}\Lambda\cdot\nabla -k^2\right] H &= f(x) & \text{ in } \Omega,\\
# H&=0 & \text{ on } \Gamma_D,\\
# H|_{\Gamma_L}&=H|_{\Gamma_R},&\\
# \end{aligned}\right.
# ```
# 
# # Numerical scheme
# 
# Similar to the previous tutorials, we can construct the weak form for this PDE: 
#
# ```math
# a(u,v) = \int_\Omega \left[\nabla(\Lambda v)\cdot\frac{1}{\varepsilon(x)}\Lambda\nabla u-k^2uv\right]\mathrm{d}\Omega
# ```
#
# ```math
# b(v) = \int_\Omega vf\mathrm{d}\Omega
# ```
#
# # Setup
# 
# We import the packages that will be used, define the geometry and physics parameters. 
# 

using Gridap
using GridapGmsh
using Gridap.Fields
using Gridap.Geometry

# Geometry parameters
λ = 1.0          # Wavelength (arbitrary unit)
L = 4.0          # Width of the area
H = 6.0          # Height of the area
xc = [0 -1.0]    # Center of the cylinder
r = 1.0          # Radius of the cylinder
d_pml = 0.8      # Thickness of the PML   
# Physical parameters 
k = 2*π/λ        # Wave number 
ϵ1 = 3.0         # Relative electric permittivity for cylinder

# # Discrete Model
# 
# We import the model from the `geometry.msh` mesh file using the `GmshDiscreteModel` function defined in `GridapGmsh`. The mesh file is created with the GMSH (see the file ../assets/emscatter/MeshGenerator.jl). Note that this mesh file already contains periodic boundary information for the left and right side, and that is enough for gridap to implement the periodic boundary condition. 
# 

model = GmshDiscreteModel("../models/geometry.msh")

# # FE spaces
# 
# We use the first-order lagrangian as the finite element function space basis. The dirihlet edges are labeld with `DirichletEdges` in the mesh file. Since our problem involves complex numbers (because of PML), we need to assign the `vector_type` to be `Vector{ComplexF64}`.
# 

# Test and trial finite element function space
order = 1
reffe = ReferenceFE(lagrangian,Float64,order)
V = TestFESpace(model,reffe,dirichlet_tags=["DirichletEdges"],vector_type=Vector{ComplexF64})
U = TrialFESpace(V,[0])

# # Numerical integration
# 
# We generate the triangulation and a second-order Gaussian quadrature for the numerial integration. Note that we create a boundary triangulation from a `Source` tag for the line excitation. Generally, we do not need such additional mesh tags for the source, we can use a delta function to approximate such line source excitation. However, by generating a line mesh, we can increase the accuracy of this source excitation.
# 

# Generate triangulation and quadrature from model 
degree = 2
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)
# Source triangulation
Γ = BoundaryTriangulation(model;tags="Source")
dΓ = Measure(Γ,degree)

# # PML formulation
# 
# More detailed demonstration of the PML formulation can be found in Ref [1]. Here we first define a `s_PML` function: $s(x)=1+\mathrm{i}\sigma(u)/\omega,$ and its derivative `ds_PML`. The parameter `LH` indicates the size of the inner boundary of the PML regions. Finally, we create a function-like object `Λ` that returns the PML factors and define its derivative in gridap. 
# 

# PML parameters
R = 1e-4      # Tolerence for PML reflection 
σ = -3/4*log(R)/d_pml
LH = [L,H]

# PML coordinate streching functions
function s_PML(x,σ,k,LH,d_pml)
    u = abs.(Tuple(x)).-LH./2
    return @. ifelse(u > 0,  1+(1im*σ/k)*(u/d_pml)^2, $(1.0+0im))
end

function ds_PML(x,σ,k,LH,d_pml)
    u = abs.(Tuple(x)).-LH./2
    ds = @. ifelse(u > 0, (2im*σ/k)*(1/d_pml)^2*u, $(0.0+0im))
    return ds.*sign.(Tuple(x))
end

struct Λ<:Function
    σ::Float64
    k::Float64
    LH::Vector{Float64}
    d_pml::Float64
end

function (Λf::Λ)(x)
    s_x,s_y = s_PML(x,Λf.σ,Λf.k,Λf.LH,Λf.d_pml)
    return VectorValue(1/s_x,1/s_y) 
end

Fields.∇(Λf::Λ) = x->TensorValue{2,2,ComplexF64}(-(Λf(x)[1])^2*ds_PML(x,Λf.σ,Λf.k,Λf.LH,Λf.d_pml)[1],0,0,-(Λf(x)[2])^2*ds_PML(x,Λf.σ,Λf.k,Λf.LH,Λf.d_pml)[2])


# # Weak form
# 
# In the mesh file, we labeled the cylinder region with `Cylinder` to distinguish it from other regions. Using this tag, we can assign material properties correspondingly (basically a function with different value in different regions). The weak form is very similar to its mathematical form in gridap.
# 

labels = get_face_labeling(model)
dimension = num_cell_dims(model)
tags = get_face_tag(labels,dimension)
const cylinder_tag = get_tag_from_name(labels,"Cylinder")

function ξ(tag)
    if tag == cylinder_tag
        return 1/ϵ1
    else
        return 1.0
    end
end

a(u,v) = (∇.*(Λ(σ,k,LH,d_pml)*v))⊙((ξ∘CellField(tags,Ω))*(Λ(σ,k,LH,d_pml).*∇(u))) - (k^2*(v*u))

# Source term (Line source)
b(v) = ∫(v)*dΓ


# # Solver phase
# 
# We can assemble the matrix and source term vector with the weak form and then solve for the field.
# 

A_mat = assemble_matrix(U,V) do u,v
    ∫( a(u,v) )dΩ
end
b_vec = assemble_vector(b_Ω,V)
u_vec = A_mat\b_vec
uh = FEFunction(U,u_vec)

# # Analytical solution
# 
# In 2D cylinder coordinates, we can expand the plane wave in terms of Bessel functions: Plane wave: 
#
# ```math
# H_0=\sum_m i^mJ_m(kr)e^{im\theta},
# ```
#
# where $m=0,\pm 1,\pm 2,\dots$ and $J_m(z)$ is the Bessel function of the fisrt kind. 
# 
# Consider the $m$-th component, the incident part is: 
#
# ```math
# H_{inc}=J_m(kr),
# ```
#
# the scattered field is : 
#
# ```math
# H_1=\alpha_mH_m^1(kr),
# ```
#
# and the fields inside the cylinder is: 
#
# ```math
# H_2=\beta_mJ_m(nkr).
# ```
# 
# Here, $H_m^1$ is the Bessel function of the third kind (Hankel function of the first kind) that denotes a outgoing cylindrical wave, $n=\sqrt{\varepsilon}$ is the refractive index. 
# 
# Applyin the boundary condition: 
#
# ```math
# H_{inc}+H_1=H_2|_{r=R},
# ```
#
# ```math
# \frac{\partial H_{inc}}{\partial r}+\frac{\partial H_1}{\partial r}=\frac{1}{\epsilon}\frac{\partial H_2}{\partial r}|_{r=R}.
# ```
# 
# We get: 
#
# ```math
# \alpha_m=\frac{J_m(nkR)J_m(kR)^\prime-\frac{1}{n}J_m(kR)J_m(nkR)^\prime}{\frac{1}{n}H_m^1(kR)J_m(nkr)^\prime-J_m(nkr)H_m^1(kr)^\prime},
# ```
# 
# ```math
# \beta_m = \frac{H_m^1(kR)J_m(kR)^\prime-J_m(kR)H_m^1(kR)^\prime}{\frac{1}{n}J_m(nkR)^\prime H_m^1(kR)-J_m(nkR)H_m^1(kR)^\prime},
# ```
# 
# where $^\prime$ denotes the derivative, and it is obtained by the recurrent relation of the Bessel functions: 
#
# ```math
# Y_m(z)^\prime=\frac{Y_{m-1}(z)-Y_{m+1}(z)}{2}
# ```
#
# with $Y_m$ denotes any Bessel functions.
#
# 
# Finally, the analytical field is ($2k$ comes from the unit source excitation):
# ```math
# H(r>R)=\frac{1}{2k}\sum_m\left[\alpha_mi^mH_m^1(kr)+J_m(kr)\right]e^{im\theta}
# ```
#
# ```math
# H(r<=R)=\frac{1}{2k}\sum_m\beta_mi^mJ_m(nkr)e^{im\theta}
# ```
# 

using SpecialFunctions
dbesselj(m,z) = (besselj(m-1,z)-besselj(m+1,z))/2
dhankelh1(m,z)= (hankelh1(m-1,z)-hankelh1(m+1,z))/2
α(m,n,z) = (besselj(m,n*z)*dbesselj(m,z)-1/n*besselj(m,z)*dbesselj(m,n*z))/(1/n*hankelh1(m,z)*dbesselj(m,n*z)-besselj(m,n*z)*dhankelh1(m,z))
β(m,n,z) = (hankelh1(m,z)*dbesselj(m,z)-besselj(m,z)*dhankelh1(m,z))/(1/n*dbesselj(m,n*z)*hankelh1(m,z)-besselj(m,n*z)*dhankelh1(m,z))

function H_t(x,xc,r,ϵ,λ)
    n = √ϵ
    k = 2*π/λ
    θ = angle(x[1]-xc[1]+1im*(x[2]-xc[2]))+π
    M = 20
    H0 = 0
    if norm([x[1]-xc[1],x[2]-xc[2]])<=r
        for m=-M:M
            H0 += β(m,n,k*r)*cis(m*θ)*besselj(m,n*k*norm([x[1]-xc[1],x[2]-xc[2]]))
        end
    else
        for m=-M:M
            H0 += α(m,n,k*r)*cis(m*θ)*hankelh1(m,k*norm([x[1]-xc[1],x[2]-xc[2]]))+cis(m*θ)*besselj(m,k*norm([x[1]-xc[1],x[2]-xc[2]]))
        end
    end
    return 1im/(2*k)*H0
end
# Construct the analytical solution with finite element functions
a_theory(u,v) = ∫(u*v)*dΩ
b_theory(v) = ∫(v*(x->H_t(x,xc,r,ϵ1,λ)))*dΩ
A_t= assemble_matrix(a_theory,U,V)
b_t = assemble_vector(b_theory,V)
u_t = A_t\b_t
uh_t = FEFunction(U,u_t)

# # Output and compare results
# 
# The simulated field is shown below. We can also see that the difference between the simulated fields and the analytical solution is almost zero near the cylidner region, which validates the simulation. The difference is larger far away from the cylinder center, that is because we only used M=20 to approximate the sum, the larger M, the smaller difference.
# ![](../assets/emscatter/Results.png)

# Save to file and view
writevtk(Ω,"demo",cellfields=["Real"=>real(uh),
        "Imag"=>imag(uh),
        "Norm"=>abs2(uh),
        "Real_t"=>real(uh_t),
        "Imag_t"=>imag(uh_t),
        "Norm_t"=>abs2(uh_t),
        "Difference"=>abs2(uh_t-uh)])
#
# ## References
#
# [1] A. F. Oskooi, L. Zhang, Y. Avniel, and S. G. Johnson, “The failure of perfectly matched layers, and towards their redemption by adiabatic absorbers,” Optics Express, vol. 16, pp. 11376–11392, July 2008.