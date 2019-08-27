using Literate

repo_src = joinpath(@__DIR__,"..","src")
notebooks_dir = joinpath(@__DIR__,"..","notebooks")

Literate.notebook(joinpath(repo_src,"t001_poisson.jl"), notebooks_dir; documenter=false, execute=false)
