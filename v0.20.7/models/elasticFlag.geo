// Gmsh project created on Fri Mar 20 14:01:03 2020
SetFactory("OpenCASCADE");

// Define Geometry
Rectangle(1) = {0, 0, 0, 2.5, 0.41, 0};
Disk(2) = {0.2, 0.2, 0, 0.05, 0.05};
Rectangle(3) = {0.2, 0.19, 0, 0.4, 0.02, 0};
BooleanDifference{ Surface{1}; Delete; }{ Surface{2,3}; }
BooleanDifference{ Surface{3}; Delete; }{ Surface{2}; Delete;}
Coherence;

// Define mesh sizes
Transfinite Curve {16, 15} = 2 Using Progression 1;
Transfinite Curve {7} = 3 Using Progression 1;
Transfinite Curve {14, 13, 12} = 20 Using Progression 1;
Transfinite Curve {9} = 20 Using Progression 1;
Transfinite Curve {10} = 10 Using Progression 1;
Transfinite Curve {11, 8} = 60 Using Progression 1;

// Define Physical groups
Physical Surface("fluid") = {1};
Physical Surface("solid") = {3};
Physical Curve("inlet") = {9};
Physical Curve("outlet") = {10};
Physical Curve("noSlip") = {11, 8};
Physical Curve("fixed") = {16, 15};
Physical Curve("cylinder") = {12};
Physical Curve("interface") = {14, 7, 13};
Physical Point("fixed") = {13, 15, 14};
Physical Point("interface") = {8, 7};
Physical Point("inlet") = {9, 11};
Physical Point("outlet") = {12, 10};
