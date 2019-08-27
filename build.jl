using Literate

Literate.markdown("src/t001_poisson.jl", "markdown"; documenter=false)
Literate.notebook("src/t001_poisson.jl", "notebooks"; documenter=false, execute=false)
