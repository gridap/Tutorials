#
# In this tutorial, we will learn
#
#    -  How the `DiscreteModel` and its components work.
#    -  How to extract topological information from a `GridTopology`.
#    -  How to extract geometrical information from a `Grid`.
#    -  How periodicity is handled in Gridap, and the difference between nodes and vertices.
#    -  How to create a periodic model from scratch, use the example of a Mobius strip.
#    -  How to create and manipulate `FaceLabeling` objects, which are used to handle boundary conditions.
#
# ## Required Packages

using Gridap
using Gridap.Geometry, Gridap.ReferenceFEs, Gridap.Arrays
using Plots

#
# ## Table of Contents
# 1. Utility Functions
# 2. The DiscreteModel Structure
# 3. Working with Topology
# 4. Geometric Mappings
# 5. High-order Grids
# 6. Periodicity in Gridap
# 7. FaceLabelings
#
# ## 1. Utility Functions
# We begin by defining helper functions that will be essential throughout this tutorial.
# These functions help us visualize and work with our mesh structures.

# Convert a `CartesianDiscreteModel` to an `UnstructuredDiscreteModel` for more generic handling.
function cartesian_model(args...; kwargs...)
  UnstructuredDiscreteModel(CartesianDiscreteModel(args...; kwargs...))
end

# Visualization function to plot nodes with their IDs. Input:
# - node_coords: Array of node coordinates.
# - node_ids: Array of corresponding node IDs.
function plot_node_numbering(node_coords, node_ids)
  x = map(c -> c[1], node_coords)
  y = map(c -> c[2], node_coords)
  a = text.(node_ids, halign=:left, valign=:bottom)
  scatter(x, y, series_annotations = a, legend=false)
  hline!(unique(x), linestyle=:dash, color=:grey)
  vline!(unique(y), linestyle=:dash, color=:grey)
end

# Overloaded method to plot node numbering directly from a model.
# This function extracts the necessary information from the model and calls the base plotting function.
function plot_node_numbering(model)
  D = num_cell_dims(model)
  topo = get_grid_topology(model)
  node_coords = Geometry.get_node_coordinates(model)
  cell_node_ids = get_cell_node_ids(model)
  cell_vertex_ids = Geometry.get_faces(topo, D, 0)

  node_to_vertex = zeros(Int, length(node_coords))
  for (nodes,vertices) in zip(cell_node_ids, cell_vertex_ids)
    node_to_vertex[nodes] .= vertices
  end
  
  plot_node_numbering(node_coords, node_to_vertex)
end

#
# ## 2. The DiscreteModel Structure
#
# The `DiscreteModel` in Gridap is a fundamental structure that represents a discretized 
# computational domain. It consists of three main components, each serving a specific purpose:
#
#   - The `GridTopology`: Defines the connectivity of the mesh elements
#     * Stores how vertices, edges, faces, and cells are connected
#     * Enables neighbor queries and traversal of the mesh
#     * Pure topological information, no geometric data
#
#   - The `Grid`: Contains the geometric information of the mesh
#     * Stores coordinates of mesh nodes
#     * Provides mappings between reference and physical elements
#     * Handles curved elements and high-order geometries
#
#   - The `FaceLabeling`: Manages mesh labels and markers
#     * Identifies boundary regions
#     * Tags different material regions
#     * Essential for applying boundary conditions
#
# ### Key Concept: Nodes vs. Vertices
#
# A very important distinction in Gridap is between nodes and vertices:
#
#   - **Vertices** (Topological entities):
#     * 0-dimensional entities in the `GridTopology`
#     * Define the connectivity of the mesh
#     * Used for neighbor queries and mesh traversal
#     * Number of vertices depends only on topology
#
#   - **Nodes** (Geometrical entities):
#     * Control points stored in the `Grid`
#     * Define the geometry of elements
#     * Used for interpolation and mapping
#     * Number of nodes depends on the geometric order
#
# While nodes and vertices often coincide in simple meshes, they differ in two important cases:
# 1. Periodic meshes: Where multiple nodes may correspond to the same vertex
# 2. High-order meshes: Where additional nodes are needed to represent curved geometries
#
# ## 3. Working with Topology
#
# Let's explore how to extract and work with topological information. We'll start by creating
# a simple 3x3 cartesian mesh:

model = cartesian_model((0,1,0,1),(3,3))

# First, let's get the topology component and count the mesh entities:

topo = get_grid_topology(model)

n_vertices = num_faces(topo,0)  # Number of vertices (0-dimensional entities)
n_edges = num_faces(topo,1)     # Number of edges (1-dimensional entities)
n_cells = num_faces(topo,2)     # Number of cells (2-dimensional entities)

# ### Connectivity Queries
# Gridap provides various methods to query the connectivity between different mesh entities.
# Here are some common queries:

# Get vertices of each cell (cell → vertex connectivity)
cell_to_vertices = get_faces(topo,2,0)

# Get edges of each cell (cell → edge connectivity)
cell_to_edges = get_faces(topo,2,1)

# Get cells adjacent to each edge (edge → cell connectivity)
edge_to_cells = get_faces(topo,1,2)

# Get vertices of each edge (edge → vertex connectivity)
edge_to_vertices = get_faces(topo,1,0)

# ### Advanced Connectivity: Finding Cell Neighbors
#
# Finding cells that share entities (like vertices or edges) requires more work.
# A direct query for cell-to-cell connectivity returns an identity map:

cell_to_cells = get_faces(topo,2,2)  # Returns identity table

# To find actual cell neighbors, we need to traverse through lower-dimensional entities.
# Here's a utility function that builds a face-to-face connectivity graph:

function get_face_to_face_graph(topo,Df)
  n_faces = num_faces(topo,Df)
  face_to_vertices = get_faces(topo,Df,0)  # Get vertices of each face
  vertex_to_faces = get_faces(topo,0,Df)   # Get faces incident to each vertex

  face_to_face = Vector{Vector{Int}}(undef,n_faces)
  for face in 1:n_faces
    nbors = Int[]
    for vertex in face_to_vertices[face]
      append!(nbors,vertex_to_faces[vertex]) # Add incident faces
    end
    face_to_face[face] = filter(!isequal(face),unique(nbors)) # Remove self-reference and duplicates
  end

  return face_to_face
end

# Now we can find neighboring cells and edges:
cell_to_cells = get_face_to_face_graph(topo,2)  # Cells sharing vertices
edge_to_edges = get_face_to_face_graph(topo,1)  # Edges sharing vertices

#
# ## 4. Geometric Mappings
#
# The geometry of our mesh is defined by mappings from reference elements to physical space.
# Let's explore how these mappings work in Gridap:

grid = get_grid(model)

# First, we extract the basic geometric information:
cell_map = get_cell_map(grid)          # Mapping from reference to physical space
cell_to_nodes = get_cell_node_ids(grid) # Node IDs for each cell
node_coordinates = get_node_coordinates(grid) # Physical coordinates of nodes

# ### Computing Cell-wise Node Coordinates
#
# There are two ways to get the coordinates of nodes for each cell:
#
# A) Using standard Julia mapping:
cell_to_node_coords = map(nodes -> node_coordinates[nodes], cell_to_nodes)

# B) Using Gridap's lazy evaluation system (more efficient for large meshes):
cell_to_node_coords = lazy_map(Broadcasting(Reindex(node_coordinates)),cell_to_nodes)

# ### Geometric Mappings
#
# The mapping from reference to physical space is defined by cell-wise linear combination of:
# 1. Reference element shape functions (basis)
# 2. Physical node coordinates (coefficients)

cell_reffes = get_cell_reffe(grid)     # Get reference elements for each cell
cell_basis = lazy_map(get_shapefuns,cell_reffes)  # Get basis functions
cell_map = lazy_map(linear_combination,cell_to_node_coords,cell_basis)

# ## 5. High-order Grids
#
# High-order geometric representations are essential for accurately modeling curved boundaries
# and complex geometries. Let's explore this by creating a curved mesh:
#
# ### Example: Creating a Half-Cylinder
#
# First, we define a mapping that transforms our planar mesh into a half-cylinder:

function F(x)
  θ = x[1]*pi   # Map x-coordinate to angle [0,π]
  z = x[2]      # Keep y-coordinate as height
  VectorValue(cos(θ),sin(θ),z)  # Convert to cylindrical coordinates
end

# Apply the mapping to our node coordinates:
new_node_coordinates = map(F,node_coordinates)

# Create new cell mappings with the transformed coordinates:
new_cell_to_node_coords = lazy_map(Broadcasting(Reindex(new_node_coordinates)),cell_to_nodes)
new_cell_map = lazy_map(linear_combination,new_cell_to_node_coords,cell_basis)

# Create a new grid with the transformed geometry:
reffes, cell_types = compress_cell_data(cell_reffes)
new_grid = UnstructuredGrid(new_node_coordinates,cell_to_nodes,reffes,cell_types)

# Save for visualization:
writevtk(new_grid,"half_cylinder_linear")

#
# If we visualize the result, we'll notice that despite applying a curved mapping,
# our half-cylinder looks faceted. This is because we're still using linear elements
# (straight edges) to approximate the curved geometry.
#
# ### Example: High-order Elements
#
# To accurately represent curved geometries, we need high-order elements:

# Create quadratic reference elements:
order = 2  # Polynomial order
new_reffes = [LagrangianRefFE(Float64,QUAD,order)]  # Quadratic quadrilateral elements
new_cell_reffes = expand_cell_data(new_reffes,cell_types)

# Create a finite element space to handle the high-order geometry:
space = FESpace(model,new_cell_reffes)
new_cell_to_nodes = get_cell_dof_ids(space)

# Get the quadratic nodes in the reference element:
cell_dofs = lazy_map(get_dof_basis,new_cell_reffes)
cell_basis = lazy_map(get_shapefuns,new_cell_reffes)
cell_to_ref_coordinates = lazy_map(get_nodes,cell_dofs)

# Map the reference nodes to the physical space:
cell_to_phys_coordinates = lazy_map(evaluate,cell_map,cell_to_ref_coordinates)

# Create the high-order node coordinates:
new_n_nodes = maximum(maximum,new_cell_to_nodes)
new_node_coordinates = zeros(VectorValue{2,Float64},new_n_nodes)
for (cell,nodes) in enumerate(new_cell_to_nodes)
  for (i,node) in enumerate(nodes)
    new_node_coordinates[node] = cell_to_phys_coordinates[cell][i]
  end
end

# Apply our cylindrical mapping to the high-order nodes:
new_node_coordinates = map(F,new_node_coordinates)

# Create the high-order grid:
new_grid = UnstructuredGrid(new_node_coordinates,new_cell_to_nodes,new_reffes,cell_types)
writevtk(new_grid,"half_cylinder_quadratic")

# The resulting mesh now accurately represents the curved geometry of the half-cylinder,
# with quadratic elements properly capturing the curvature (despite paraview still showing 
# a linear interpolation between the nodes).

#
# ## 6. Periodicity in Gridap
#
# Periodic boundary conditions are essential in many applications, such as:
# - Modeling crystalline materials
# - Simulating fluid flow in periodic domains
# - Analyzing electromagnetic wave propagation
#
# Gridap handles periodicity through a clever approach:
# 1. In the topology: Periodic vertices are "glued" together, creating a single topological entity
# 2. In the geometry: The corresponding nodes maintain their distinct physical positions
#
# This separation between topology and geometry allows us to:
# - Maintain the correct geometric representation
# - Automatically enforce periodic boundary conditions
# - Avoid mesh distortion at periodic boundaries
#
# ### Visualizing Periodicity
#
# Let's examine how periodicity affects the mesh structure through three examples:

# 1. Standard non-periodic mesh:
model = cartesian_model((0,1,0,1),(3,3))
plot_node_numbering(model)
# ![](../assets/geometry/nodes_nonperiodic.png)

# 2. Mesh with periodicity in x-direction:
model = cartesian_model((0,1,0,1),(3,3),isperiodic=(true,false))
plot_node_numbering(model)
# ![](../assets/geometry/nodes_halfperiodic.png)

# 3. Mesh with periodicity in both directions:
model = cartesian_model((0,1,0,1),(3,3),isperiodic=(true,true))
plot_node_numbering(model)
# ![](../assets/geometry/nodes_fullperiodic.png)

# Notice how the vertex numbers (displayed at node positions) show the topological
# connectivity, while the nodes remain at their physical positions.
#
# ### Creating a Möbius Strip
#
# We'll create it by:
# 1. Starting with a rectangular mesh
# 2. Making it periodic in one direction
# 3. Adding a twist before connecting the ends
#
# #### Step 1: Create Base Mesh
#
# Start with a 3x3 cartesian mesh:

nc = (3,3)  # Number of cells in each direction
model = cartesian_model((0,1,0,1),nc)

# Extract geometric and topological information:
node_coords = get_node_coordinates(model)  # Physical positions
cell_node_ids = get_cell_node_ids(model)  # Node connectivity
cell_type = get_cell_type(model)          # Element type
reffes = get_reffes(model)                # Reference elements

# #### Step 2: Create Periodic Topology
#
# To create the Möbius strip, we need to:
# 1. Identify vertices to be connected
# 2. Reverse one edge to create the twist
# 3. Ensure proper connectivity

# Create initial vertex numbering:
np = nc .+ 1  # Number of points in each direction
mobius_ids = collect(LinearIndices(np))

# Create the twist by reversing the last row:
mobius_ids[end,:] = reverse(mobius_ids[1,:])

# Map cell nodes to the new vertex numbering:
cell_vertex_ids = map(nodes -> mobius_ids[nodes], cell_node_ids)

# #### Step 3: Clean Up Vertex Numbering
#
# The new vertex IDs aren't contiguous (some numbers are duplicated due to periodicity).
# We need to create a clean mapping:

# Find unique vertices and create bidirectional mappings:
vertex_to_node = unique(vcat(cell_vertex_ids...))
node_to_vertex = find_inverse_index_map(vertex_to_node)

# Renumber vertices to be contiguous:
cell_vertex_ids = map(nodes -> node_to_vertex[nodes], cell_vertex_ids)

# Convert to the Table format required by Gridap:
cell_vertex_ids = Table(cell_vertex_ids)

# Get coordinates for the vertices:
vertex_coords = node_coords[vertex_to_node]
polytopes = map(get_polytope,reffes)

# #### Step 4: Create the Model
#
# Now we can create our Möbius strip model:

# Create topology with periodic connections:
topo = UnstructuredGridTopology(
  vertex_coords, cell_vertex_ids, cell_type, polytopes
)

# Create grid (geometry remains unchanged):
grid = UnstructuredGrid(
  node_coords, cell_node_ids, reffes, cell_type
)

# Add basic face labels:
labels = FaceLabeling(topo)

# Combine into final model:
mobius = UnstructuredDiscreteModel(grid,topo,labels)

# Visualize the vertex numbering:
plot_node_numbering(mobius)
# ![](../assets/geometry/mobius.png)

# ## 7. FaceLabelings and boundary conditions
#
# The `FaceLabeling` component of a `DiscreteModel` is the way Gridap handles boundary conditions.
# The basic idea is that, similar to Gmsh, we classify the d-faces (cells, faces, edges, nodes) of the mesh 
# into different entities (physical groups in Gmsh terminology) which in turn have one or more 
# tags/labels associated with them.
# We can then query the `FaceLabeling` for the tags associated with a given d-face, 
# or the d-faces associated with a given tag.
#
# We will now explore ways to create and manipulate `Facelabeling` objects.
#
# ### Creating FaceLabelings
#
# The simplest way to create a blank `FaceLabeling` is to use your `GridTopology`: 
#

model = cartesian_model((0,1,0,1),(3,3))
topo = get_grid_topology(model)

labels = FaceLabeling(topo)

# The above `FaceLabeling` is by default created with 2 entities and 2 tags, associated to 
# interior and boundary d-faces respectively. The boundary facets are chosen as the ones 
# with a single neighboring cell. 
# 
# We can extract the low-level information from the `FaceLabeling` object:

tag_names = get_tag_name(labels) # Each name is a string
tag_entities = get_tag_entities(labels) # For each tag, a vector of entities
cell_to_entity = get_face_entity(labels,2) # For each cell, its associated entity
edge_to_entity = get_face_entity(labels,1) # For each edge, its associated entity
node_to_entity = get_face_entity(labels,0) # For each node, its associated entity

# It is usually more convenient to visualise it in Paraview by exporting to vtk: 

writevtk(model,"labels_basic",labels=labels)

# Another useful way to create a `FaceLabeling` is by providing a coloring for the mesh cells, 
# where each color corresponds to a different tag.
# The d-faces of the mesh will have all the tags associated to the cells that share them.

cell_to_tag = [1,1,1,2,2,3,2,2,3]
tag_to_name = ["A","B","C"]
labels_cw = Geometry.face_labeling_from_cell_tags(topo,cell_to_tag,tag_to_name)
writevtk(model,"labels_cellwise",labels=labels_cw)

# We can also create a `FaceLabeling` from a vertex filter. The resulting `FaceLabeling` will have
# only one tag, gathering the d-faces whose vertices ALL fullfill `filter(x) == true`.

vfilter(x) = abs(x[1]- 1.0) < 1.e-5
labels_vf = Geometry.face_labeling_from_vertex_filter(topo, "top", vfilter)
writevtk(model,"labels_filter",labels=labels_vf)

# `FaceLabeling` objects can also be merged together. The resulting `FaceLabeling` will have 
# the union of the tags and entities of the original ones.
# Note that this modifies the first `FaceLabeling` in place.

labels = merge!(labels, labels_cw, labels_vf)
writevtk(model,"labels_merged",labels=labels)

# ### Creating new tags from existing ones
#
# Tags in a `FaceLabeling` support all the usual set operation, i.e union, intersection, 
# difference and complementary. 

cell_to_tag = [1,1,1,2,2,3,2,2,3]
tag_to_name = ["A","B","C"]
labels = Geometry.face_labeling_from_cell_tags(topo,cell_to_tag,tag_to_name)

# Union: Takes as input a list of tags and creates a new tag that is the union of all of them.
Geometry.add_tag_from_tags!(labels,"A∪B",["A","B"])

# Intersection: Takes as input a list of tags and creates a new tag that is the intersection of all of them.
Geometry.add_tag_from_tags_intersection!(labels,"A∩B",["A","B"])

# Complementary: Takes as input a list of tags and creates a new tag that is the complementary of the union.
Geometry.add_tag_from_tags_complementary!(labels,"!A",["A"])

# Set difference: Takes as input two lists of tags (tags_include - tags_exclude) 
# and creates a new tag that contains all the d-faces that are in the first list but not in the second.
Geometry.add_tag_from_tags_setdiff!(labels,"A-B",["A"],["B"]) # set difference

writevtk(model,"labels_setops",labels=labels)

# ### FaceLabeling queries
#
# The most common way of query information from a `FaceLabeling` is to query a face mask for 
# a given tag and face dimension. If multiple tags are provided, the union of the tags is returned.

face_dim = 1
mask = get_face_mask(labels,["A","C"],face_dim) # Boolean mask
ids = findall(edges_mask_A) # Edge IDs
