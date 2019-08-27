using Documenter
using Literate

models_src = joinpath(@__DIR__,"..","models")
models_dst = joinpath(@__DIR__,"src","models")

assets_src = joinpath(@__DIR__,"..","assets")
assets_dst = joinpath(@__DIR__,"src","assets")

Sys.rm(models_dst;recursive=true,force=true)
Sys.rm(assets_dst;recursive=true,force=true)

Sys.mkdir(models_dst)
fn = "model-r1.png"
Sys.cp(joinpath(models_src,fn),joinpath(models_dst,fn))

Sys.cp(assets_src,assets_dst)

pages_dir = joinpath(@__DIR__,"src","pages")
notebooks_dir = joinpath(@__DIR__,"src","notebooks")

repo_src = joinpath(@__DIR__,"..","src")

files = ["t001_poisson"]

for file in files
  file_jl = file*".jl"
  Literate.markdown(joinpath(repo_src,file_jl), pages_dir; codefence="```julia" => "```")
  Literate.notebook(joinpath(repo_src,file_jl), notebooks_dir; documenter=false, execute=false)
end

makedocs(
    sitename = "Gridap tutorials",
    format = Documenter.HTML(),
    pages =["Introduction"=> "index.md", "1 Poisson equation" => "pages/t001_poisson.md"]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.

deploydocs(
    repo = "github.com/gridap/Tutorials.git"
)
