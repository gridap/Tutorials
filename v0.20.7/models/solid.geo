//+
Point(1) = {-1, 0, 0, 1.0};
//+
Point(2) = {-0.8, 0, 0, 1.0};
//+
Point(3) = {-0.7, 0, 0, 1.0};
//+
Point(4) = {-0.6, 0, 0, 1.0};
//+
Point(5) = {-1.2, 0, 0, 1.0};
//+
Point(6) = {-1.3, 0, 0, 1.0};
//+
Point(7) = {-1.4, 0, 0, 1.0};
//+
Point(8) = {-1, 0.2, 0, 1.0};
//+
Point(9) = {-1, 0.3, 0, 1.0};
//+
Point(10) = {-1, 0.4, 0, 1.0};
//+
Point(11) = {-1, -0.2, 0, 1.0};
//+
Point(12) = {-1, -0.3, 0, 1.0};
//+
Point(13) = {-1, -0.4, 0, 1.0};
//+
Point(14) = {0.6, 0, 0, 1.0};
//+
Point(15) = {1.1, 0, 0, 1.0};
//+
Point(16) = {0.9, 0, 0, 1.0};
//+
Point(17) = {0.6, 0.5, 0, 1.0};
//+
Point(18) = {0.6, 0.3, 0, 1.0};
//+
Point(19) = {0.6, -0.3, 0, 1.0};
//+
Point(20) = {0.3, 0, 0, 1.0};
//+
Point(21) = {0.6, -0.5, 0, 1.0};
//+
Line(1) = {10, 17};
//+
Line(2) = {13, 21};
//+
Circle(3) = {10, 1, 7};
//+
Circle(4) = {7, 1, 13};
//+
Circle(5) = {21, 14, 15};
//+
Circle(6) = {15, 14, 17};
//+
Circle(7) = {16, 14, 18};
//+
Circle(8) = {18, 14, 20};
//+
Circle(9) = {20, 14, 19};
//+
Circle(10) = {19, 14, 16};
//+
Circle(11) = {2, 1, 8};
//+
Circle(12) = {8, 1, 5};
//+
Circle(13) = {5, 1, 11};
//+
Circle(14) = {11, 1, 2};
//+
Circle(15) = {3, 1, 9};
//+
Circle(16) = {9, 1, 6};
//+
Circle(17) = {6, 1, 12};
//+
Circle(18) = {12, 1, 3};
//+
Recursive Delete {
  Point{4}; 
}
//+
Curve Loop(1) = {1, -6, -5, -2, -4, -3};
//+
Curve Loop(2) = {8, 9, 10, 7};
//+
Curve Loop(3) = {15, 16, 17, 18};
//+
Plane Surface(1) = {1, 2, 3};
//+
Curve Loop(4) = {11, 12, 13, 14};
//+
Plane Surface(2) = {3, 4};
//+
Extrude {0, 0, 0.1} {
  Surface{1}; 
}
//+
Extrude {0, 0, 0.1} {
  Surface{2}; 
}
//+
Extrude {0, 0, 0.1} {
  Surface{132}; 
}
//+
Extrude {0, 0, -0.1} {
  Surface{2}; 
}
//+
Physical Volume("material_1") = {1};
//+
Physical Volume("material_2") = {3, 2, 4};
//+
Physical Surface("surface_1") = {61, 65};
//+
Physical Curve("surface_1") = {60};
//+
Physical Curve("surface_1_c") = {26, 27, 10, 7, 59, 64};
//+
Physical Point("surface_1_c") = {50, 18, 52, 16, 57, 19};
//+
Physical Surface("surface_2") = {173, 131, 215, 161, 119, 203, 165, 123, 207, 169, 127, 211};
//+
Physical Curve("surface_2") = {168, 126, 210, 164, 122, 206, 160, 118, 202, 159, 117, 201};
//+
Physical Curve("surface_2") += {98, 97, 13, 12, 99, 11, 96, 14};
//+
Physical Point("surface_2") = {115, 8, 103, 2, 105, 11, 110, 5};
//+
Physical Curve("surface_2_c") = {140, 141, 138, 139, 183, 180, 181, 182};
//+
Physical Point("surface_2_c") = {148, 143, 138, 136, 176, 181, 171, 169};
