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

Sys.rm(pages_dir;recursive=true,force=true)
Sys.rm(notebooks_dir;recursive=true,force=true)

repo_src = joinpath(@__DIR__,"..","src")

files = [
  "t001_poisson",
  "t002_validation",
  "t003_elasticity", 
  "t0041_p_laplacian",
  "t004_hyperelasticity",
  "t005_dg_discretization"]

for file in files
  file_jl = file*".jl"
  Literate.markdown(joinpath(repo_src,file_jl), pages_dir; codefence="```julia" => "```")
  Literate.notebook(joinpath(repo_src,file_jl), notebooks_dir; documenter=false, execute=false)
end

pages = [
  "Introduction"=> "index.md",
  "1 Poisson equation" => "pages/t001_poisson.md",
  "2 Code validation" => "pages/t002_validation.md",
  "3 Linear elasticity" => "pages/t003_elasticity.md",
  "4 p-Laplacian" => "pages/t0041_p_laplacian.md",
  "5 Hyper-elasticity" => "pages/t004_hyperelasticity.md",
  "6 Poisson equation (with DG)" => "pages/t005_dg_discretization.md"]

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
