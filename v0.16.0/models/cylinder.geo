//+
SetFactory("OpenCASCADE");
Circle(1) = {0, 0, 0, 1.5, 0, 2*Pi};
//+
Circle(2) = {0, 0, 0, 1.55, 0, 2*Pi};
//+
Curve Loop(1) = {2};
//+
Curve Loop(2) = {1};
//+
Plane Surface(1) = {1, 2};
//+
Extrude {0, 0, 4.0} {
  Surface{1}; 
}
//+
Physical Volume("volume") = {1};
//+
Physical Surface("bottom") = {1};
//+
Physical Surface("top") = {4};
//+
Physical Curve("bottom_c") = {1, 2};
//+
Physical Curve("top_c") = {4, 6};
//+
Physical Point("bottom_c") = {1, 2};
//+
Physical Point("top_c") = {3, 4};
