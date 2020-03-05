using Tutorials
using Test

for (title,filename) in Tutorials.files
    @testset "$title" begin include(joinpath("../src", filename)) end
end # module
