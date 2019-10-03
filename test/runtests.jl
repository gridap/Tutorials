using Tutorials

# We need a macro here...

module t001_poisson
using Test
@testset "t001_poisson" begin include("../src/t001_poisson.jl") end
end # module

module t002_validation
using Test
@testset "t002_validation" begin include("../src/t002_validation.jl") end
end # module

module t003_elasticity
using Test
@testset "t003_elasticity" begin include("../src/t003_elasticity.jl") end
end # module

module t0041_p_laplacian
using Test
@testset "t0041_p_laplacian" begin include("../src/t0041_p_laplacian.jl") end
end # module

module t004_hyperelasticity
using Test
@testset "t004_hyperelasticity" begin include("../src/t004_hyperelasticity.jl") end
end # module

module t005_dg_discretization
using Test
@testset "t005_dg_discretization" begin include("../src/t005_dg_discretization.jl") end
end # module

module t007_darcy
using Test
@testset "t007_darcy" begin include("../src/t007_darcy.jl") end
end # module

module t008_inc_navier_stokes
using Test
@testset "t008_inc_navier_stokes" begin include("../src/t008_inc_navier_stokes.jl") end
end # module

