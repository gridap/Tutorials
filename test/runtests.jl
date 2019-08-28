using Tutorials

# We need a macro here...

module t001_poisson
using Test
@testset "t001_poisson" begin include("../src/t001_poisson.jl") end
end # module

module t002_elasticity
using Test
@testset "t002_elasticity" begin include("../src/t002_elasticity.jl") end
end # module

module t003_hyperelasticity
using Test
@testset "t003_hyperelasticity" begin include("../src/t003_hyperelasticity.jl") end
end # module
