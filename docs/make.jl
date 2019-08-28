using Documenter
using Literate

models_src = joinpath(@__DIR__,"..","models")
models_dst = joinpath(@__DIR__,"src","models")

assets_src = joinpath(@__DIR__,"..","assets")
assets_dst = joinpath(@__DIR__,"src","assets")

Sys.rm(models_dst;recursive=true,force=true)
Sys.rm(assets_dst;recursive=true,force=true)

Sys.cp(models_src,models_dst)
Sys.cp(assets_src,assets_dst)

pages_dir = joinpath(@__DIR__,"src","pages")
notebooks_dir = joinpath(@__DIR__,"src","notebooks")

repo_src = joinpath(@__DIR__,"..","src")

files = ["t001_poisson","t002_elasticity", "t003_hyperelasticity"]

for file in files
  file_jl = file*".jl"
  Literate.markdown(joinpath(repo_src,file_jl), pages_dir; codefence="```julia" => "```")
  Literate.notebook(joinpath(repo_src,file_jl), notebooks_dir; documenter=false, execute=false)
end

pages = [
  "Introduction"=> "index.md",
  "1 Poisson equation" => "pages/t001_poisson.md",
  "2 Linear elasticity" => "pages/t002_elasticity.md",
  "3 Hyper-elasticity" => "pages/t003_hyperelasticity.md"]

makedocs(
    sitename = "Gridap tutorials",
    format = Documenter.HTML(),
    pages = pages
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.

deploydocs(
    repo = "github.com/gridap/Tutorials.git"
)
