using Documenter
using Literate
using Tutorials

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

for (title,filename) in Tutorials.files
  Literate.markdown(joinpath(repo_src,filename), pages_dir; codefence="```julia" => "```")
  Literate.notebook(joinpath(repo_src,filename), notebooks_dir; documenter=false, execute=false)
end

pages = ["Introduction"=> "index.md"]

for (i,(title,filename)) in enumerate(Tutorials.files)
    ordered_title = string(i, " ", title)
    path_to_markdown_file = joinpath("pages",string(splitext(filename)[1],".md"))
    push!(pages, (ordered_title=>path_to_markdown_file))
end

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
