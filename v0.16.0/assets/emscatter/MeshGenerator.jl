using Gmsh
import Gmsh: gmsh
function MeshGenerator(L,H,xc,r,hs,d_pml,lc)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.clear()
    gmsh.model.add("geometry")

    # Add points
    gmsh.model.geo.addPoint(-L/2-d_pml, -H/2-d_pml, 0, lc, 1)
    gmsh.model.geo.addPoint( L/2+d_pml, -H/2-d_pml, 0, lc, 2)
    gmsh.model.geo.addPoint( L/2+d_pml,  hs       , 0, lc, 3)
    gmsh.model.geo.addPoint(-L/2-d_pml,  hs       , 0, lc, 4)
    gmsh.model.geo.addPoint( L/2+d_pml,  H/2+d_pml, 0, lc, 5)
    gmsh.model.geo.addPoint(-L/2-d_pml,  H/2+d_pml, 0, lc, 6)
    gmsh.model.geo.addPoint( xc[1]-r  ,  xc[2]    , 0, lc, 7)
    gmsh.model.geo.addPoint( xc[1]    ,  xc[2]    , 0, lc, 8)
    gmsh.model.geo.addPoint( xc[1]+r  ,  xc[2]    , 0, lc, 9)
    # Add lines
    gmsh.model.geo.addLine( 1,  2,  1)
    gmsh.model.geo.addLine( 2,  3,  2)
    gmsh.model.geo.addLine( 3,  4,  3)
    gmsh.model.geo.addLine( 1,  4,  4)
    gmsh.model.geo.addLine( 3,  5,  5)
    gmsh.model.geo.addLine( 5,  6,  6)
    gmsh.model.geo.addLine( 4,  6,  7)
    gmsh.model.geo.addCircleArc( 7, 8, 9, 8)
    gmsh.model.geo.addCircleArc( 9, 8, 7, 9)
    # Construct curve loops and surfaces 
    gmsh.model.geo.addCurveLoop([1, 2, 3, -4], 1)
    gmsh.model.geo.addCurveLoop([5, 6,-7, -3], 2)
    gmsh.model.geo.addCurveLoop([8, 9], 3)
    gmsh.model.geo.addPlaneSurface([1,3], 1)
    gmsh.model.geo.addPlaneSurface([2], 2)
    gmsh.model.geo.addPlaneSurface([3], 3)
    # Physical groups
    #gmsh.model.addPhysicalGroup(0, [1,2,3,4,5,6], 1)
    #gmsh.model.setPhysicalName(0, 1, "DirichletNodes")
    gmsh.model.addPhysicalGroup(1, [1,6], 2)
    gmsh.model.setPhysicalName(1, 2, "DirichletEdges")
    gmsh.model.addPhysicalGroup(0, [7,9], 3)
    gmsh.model.setPhysicalName(0, 3, "CylinderNodes")
    gmsh.model.addPhysicalGroup(1, [8,9], 4)
    gmsh.model.setPhysicalName(1, 4, "CylinderEdges")
    gmsh.model.addPhysicalGroup(2, [3], 5)
    gmsh.model.setPhysicalName(2, 5, "Cylinder")
    gmsh.model.addPhysicalGroup(2, [1,2], 6)
    gmsh.model.setPhysicalName(2, 7, "Air")
    gmsh.model.addPhysicalGroup(1, [3], 7)
    gmsh.model.setPhysicalName(1, 7, "Source")
    gmsh.model.geo.synchronize()
    
    gmsh.model.mesh.setPeriodic(1, [2], [4],
                            [1, 0, 0, L+2*d_pml, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    gmsh.model.mesh.setPeriodic(1, [5], [7],
                            [1, 0, 0, L+2*d_pml, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    
    # We can then generate a 2D mesh...
    gmsh.model.mesh.generate(2)
    # ... and save it to disk
    gmsh.write("geometry.msh")
    gmsh.finalize()
end

# Geometry parameters
λ = 1.0          # Wavelength (arbitrary unit)
L = 4.0          # Width of the area
H = 6.0          # Height of the area
xc = [0 -1.0]    # Center of the cylinder
r = 1.0          # Radius of the cylinder
hs = 2.0         # y-position of the source (plane wave)
d_pml = 0.8      # Thickness of the PML

resol = 20.0      # Number of points per wavelength
lc = λ/resol      # Characteristic length

MeshGenerator(L,H,xc,r,hs,d_pml,lc)