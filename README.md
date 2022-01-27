# Gridap tutorials <img src="https://github.com/gridap/Gridap.jl/blob/master/images/color-logo-only.png" width="40" title="Gridap logo">

*Start solving PDEs in Julia*


| **Documentation** |
|:------------ |
| [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://gridap.github.io/Tutorials/stable) [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://gridap.github.io/Tutorials/dev)  |
|**Build Status** |
| [![Build Status](https://github.com/gridap/Tutorials/workflows/CI/badge.svg?branch=master)](https://github.com/gridap/Tutorials/actions?query=workflow%3ACI) |
| **Community** |
| [![Join the chat at https://gitter.im/Gridap-jl/community](https://badges.gitter.im/Gridap-jl/community.svg)](https://gitter.im/Gridap-jl/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) |
| **Citation** |
| [![DOI](https://joss.theoj.org/papers/10.21105/joss.02520/status.svg)](https://doi.org/10.21105/joss.02520) |




## What

This repo contains a set of tutorials to learn how to solve partial differential equations (PDEs) in Julia with the [Gridap.jl](https://github.com/gridap/Gridap.jl) library.

The tutorials are available in two formats:

- As jupyter notebooks, allowing an interactive learning experience. **This is the recommended way to follow the tutorials**

- As HTML pages, allowing a rapid access into the material without the need of any setup.

## How

Visit one of the following pages, depending of your needs, and start enjoying!

- [**STABLE**](https://gridap.github.io/Tutorials/stable) &mdash; **Tutorials for the most recently tagged version of Gridap.jl.**
- [**DEVEL**](https://gridap.github.io/Tutorials/dev) &mdash; *Tutorials for the in-development version of Gridap.jl.*

## Generating tutorials locally (only if you intend to contribute to the tutorials as a developer)

If you want to contribute to the tutorials, e.g., to make changes in their sources, you might need to generate (render) them locally to see whether the changes in the sources produce the expected outcome in the output (i.e., Jupyter notebooks + HTML pages). To this end, you have to follow the following instructions once:

```
julia --project=docs   # From the Unix shell, located at the root of Tutorials repo 
develop .              # From the Julia package manager prompt
instantiate            # "" 
build                  # "" 
exit()                 # From the Julia REPL
```

and then, each time that you perform a change on the tutorial sources, you have to execute the following command:

```
julia --project=docs docs/make.jl # From the Unix shell, located at the root of Tutorials repo 
```

to generate the tutorials. The files generated are available at `Tutorials/docs/build/`. 


## Gridap community

Join to our [gitter](https://gitter.im/Gridap-jl/community) chat to ask questions and interact with the Gridap community.

## How to cite Gridap

In order to give credit to the `Gridap` contributors, we simply ask you to cite the refence below in any publication in which you have made use of `Gridap` packages:

```
@article{Badia2020,
  doi = {10.21105/joss.02520},
  url = {https://doi.org/10.21105/joss.02520},
  year = {2020},
  publisher = {The Open Journal},
  volume = {5},
  number = {52},
  pages = {2520},
  author = {Santiago Badia and Francesc Verdugo},
  title = {Gridap: An extensible Finite Element toolbox in Julia},
  journal = {Journal of Open Source Software}
}
```

## Contact


Please, contact the project administrators, [Santiago Badia](mailto:santiago.badia@monash.edu) and [Francesc Verdugo](mailto:fverdugo@cimne.upc.edu), for further questions about licenses and terms of use.


