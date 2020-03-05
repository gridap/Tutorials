using Tutorials
using Test

for (title,filename) in Tutorials.files
    let
        @testset "$title" begin include(joinpath("../src", filename)) end
    end
end # module
