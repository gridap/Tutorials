// Gmsh project created on Fri Mar 20 14:01:03 2020
SetFactory("OpenCASCADE");

// Define Geometry
Rectangle(1) = {0, 0, 0, 2.2, 0.41, 0};
Disk(2) = {0.2, 0.2, 0, 0.05, 0.05};
BooleanDifference{ Surface{1}; Delete; }{ Surface{2}; Delete; }
Coherence;

// Define mesh sizes
Transfinite Curve {7} = 40 Using Progression 1;
Transfinite Curve {5} = 40 Using Progression 1;
Transfinite Curve {8} = 20 Using Progression 1;
Transfinite Curve {6, 9} = 80 Using Progression 1;

// Define Physical groups//+
Physical Surface("fluid", 1) = {1};
Physical Curve("inlet", 2) = {7};
Physical Curve("outlet", 3) = {8};
Physical Curve("noslip", 4) = {6, 9};
Physical Curve("cylinder", 5) = {5};
Physical Point("noslip", 4) = {6, 8, 9, 7};
Physical Point("cylinder", 5) = {5};
