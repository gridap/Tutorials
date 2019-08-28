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

module t004_hyperelasticity
using Test
@testset "t004_hyperelasticity" begin include("../src/t004_hyperelasticity.jl") end
end # module
