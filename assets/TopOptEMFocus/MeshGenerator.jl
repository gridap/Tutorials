using Gmsh
import Gmsh: gmsh

struct RecCirGeometry
    L::Float64           # Length of the normal region  
    h1::Float64          # Height of normal region
    h2::Float64          # Height of the region above source
    rt::Float64          # Radius of the target location
    rd::Float64          # Radius of design domain
    rs::Float64          # Radius of smallest distance circle
    dpml::Float64        # Thickness of the PML
    # Characteristic length (controls the resolution, smaller the finer)
    l1::Float64          # Normal region
    l2::Float64          # Design domain
end


function MeshGenerator(geo_param::RecCirGeometry, meshfile_name::String)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.clear()
    gmsh.model.add("geometry")

    # Add points
    gmsh.model.geo.addPoint(-geo_param.L/2-geo_param.dpml, -geo_param.dpml, 0, geo_param.l1, 1)
    gmsh.model.geo.addPoint( geo_param.L/2+geo_param.dpml, -geo_param.dpml, 0, geo_param.l1, 2)
    gmsh.model.geo.addPoint( geo_param.L/2+geo_param.dpml,  geo_param.h1, 0, geo_param.l1, 3)
    gmsh.model.geo.addPoint( geo_param.L/2+geo_param.dpml,  geo_param.h1+geo_param.h2+geo_param.dpml, 0, geo_param.l1, 4)
    gmsh.model.geo.addPoint(-geo_param.L/2-geo_param.dpml,  geo_param.h1+geo_param.h2+geo_param.dpml, 0, geo_param.l1, 5)
    gmsh.model.geo.addPoint(-geo_param.L/2-geo_param.dpml,  geo_param.h1, 0, geo_param.l1, 6)
    gmsh.model.geo.addPoint(0,  geo_param.h1/2, 0, geo_param.l2, 7)
    gmsh.model.geo.addPoint(-geo_param.rs,  geo_param.h1/2, 0, geo_param.l2, 8)
    gmsh.model.geo.addPoint( geo_param.rs,  geo_param.h1/2, 0, geo_param.l2, 9)
    gmsh.model.geo.addPoint(-geo_param.rd,  geo_param.h1/2, 0, geo_param.l2, 10)
    gmsh.model.geo.addPoint( geo_param.rd,  geo_param.h1/2, 0, geo_param.l2, 11)
    # Add lines
    gmsh.model.geo.addLine( 1,  2,  1)
    gmsh.model.geo.addLine( 2,  3,  2)
    gmsh.model.geo.addLine( 3,  4,  3)
    gmsh.model.geo.addLine( 4,  5,  4)
    gmsh.model.geo.addLine( 6,  5,  5)
    gmsh.model.geo.addLine( 1,  6,  6)
    gmsh.model.geo.addLine( 6,  3,  7)
    gmsh.model.geo.addCircleArc( 8, 7, 9, 8)
    gmsh.model.geo.addCircleArc( 9, 7, 8, 9)
    gmsh.model.geo.addCircleArc(10, 7,11,10)
    gmsh.model.geo.addCircleArc(11, 7,10,11)
    # Construct curve loops and surfaces 
    gmsh.model.geo.addCurveLoop([8, 9], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)
    gmsh.model.geo.addCurveLoop([10, 11], 2)
    gmsh.model.geo.addPlaneSurface([2, 1], 2)
    gmsh.model.geo.addCurveLoop([1, 2, -7,-6], 4)
    gmsh.model.geo.addPlaneSurface([4, 2], 4)
    gmsh.model.geo.addCurveLoop([3, 4,-5, 7], 5)
    gmsh.model.geo.addPlaneSurface([5], 5)
    # Physical groups
    gmsh.model.addPhysicalGroup(0, [1, 2, 4, 5], 1)
    gmsh.model.setPhysicalName(0, 1, "DirichletNodes")
    gmsh.model.addPhysicalGroup(1, [1, 4], 2)
    gmsh.model.setPhysicalName(1, 2, "DirichletEdges")
    gmsh.model.addPhysicalGroup(0, [8, 9, 10, 11], 3)
    gmsh.model.setPhysicalName(0, 3, "DesignNodes")
    gmsh.model.addPhysicalGroup(1, [8, 9, 10, 11], 4)
    gmsh.model.setPhysicalName(1, 4, "DesignEdges")
    gmsh.model.addPhysicalGroup(2, [2], 5)
    gmsh.model.setPhysicalName(2, 5, "Design")
    gmsh.model.addPhysicalGroup(2, [1], 6)
    gmsh.model.setPhysicalName(2, 6, "Center")
    gmsh.model.addPhysicalGroup(2, [3,4,5], 7)
    gmsh.model.setPhysicalName(2, 7, "Air")
    gmsh.model.addPhysicalGroup(1, [7], 9)
    gmsh.model.setPhysicalName(1, 9, "Source")
    gmsh.model.geo.synchronize()
    
    # Set periodic mesh on the left and right side
    gmsh.model.mesh.setPeriodic(1, [2], [6],
                            [1, 0, 0, geo_param.L+2*geo_param.dpml, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    gmsh.model.mesh.setPeriodic(1, [3], [5],
                            [1, 0, 0, geo_param.L+2*geo_param.dpml, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    # We can then generate a 2D mesh...
    gmsh.model.mesh.generate(2)
    
    

    # ... and save it to disk
    gmsh.write(meshfile_name)
    gmsh.finalize()
end

L = 600
h1 = 600
h2 = 200
rt = 150
rd = 100
rs = 10

dpml = 300

l1 = 20
l2 = 1

meshfile = "RecCirGeometry.msh"
geo_param = RecCirGeometry(L, h1, h2, rt, rd, rs, dpml, l1, l2)
MeshGenerator(geo_param, meshfile)