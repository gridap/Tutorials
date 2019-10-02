using Literate

repo_src = joinpath(@__DIR__,"..","src")
notebooks_dir = joinpath(@__DIR__,"..","notebooks")

files = []

files = [
  "t001_poisson"=>"t001_poisson",
  "t002_validation"=>"t002_code_validation",
  "t003_elasticity"=>"t003_linear_elasticity", 
  "t0041_p_laplacian"=>"t004_p_laplacian", 
  "t004_hyperelasticity"=>"t005_hyperelasticity",
  "t005_dg_discretization"=> "t006_poisson_with_dg",
  "t007_darcy"=>"t007_darcy_with_rt",
  "t008_inc_navier_stokes"=>"t008_inc_navier_stokes"]

Sys.rm(notebooks_dir;recursive=true,force=true)
for (file,name) in files
  file_jl = file*".jl"
  Literate.notebook(joinpath(repo_src,file_jl), notebooks_dir; name=name, documenter=false, execute=false)
end
