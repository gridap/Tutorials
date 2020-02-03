//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {1.5, 0, 0, 1.0};
//+
Point(3) = {1.5, 1, 0, 1.0};
//+
Point(4) = {0, 1, 0, 1.0};
//+
Point(5) = {0.095, 0.2, 0, 1.0};
//+
Point(6) = {0.63, 0.2, 0, 1.0};
//+
Point(7) = {0.3625, 0.675, 0, 1.0};
//+
Point(8) = {0.58, 0.5, 0, 1.0};
//+
Point(9) = {0.94, 0.5, 0, 1.0};
//+
Point(10) = {0.94, 0.86, 0, 1.0};
//+
Point(11) = {0.58, 0.86, 0, 1.0};
//+
Point(12) = {1, 0.39, 0, 1.0};
//+
Point(13) = {1.19, 0.39, 0, 1.0};
//+
Point(14) = {1.38, 0.39, 0, 1.0};
//+
Point(15) = {1.19, 0.2, 0, 1.0};
//+
Point(16) = {1.19, 0.58, 0, 1.0};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 1};
//+
Line(5) = {5, 7};
//+
Line(6) = {7, 6};
//+
Line(7) = {6, 5};
//+
Line(8) = {8, 11};
//+
Line(9) = {11, 10};
//+
Line(10) = {10, 9};
//+
Line(11) = {9, 8};
//+
Circle(12) = {12, 13, 16};
//+
Circle(13) = {16, 13, 14};
//+
Circle(14) = {14, 13, 15};
//+
Circle(15) = {15, 13, 12};
//+
Curve Loop(1) = {3, 4, 1, 2};
//+
Curve Loop(2) = {5, 6, 7};
//+
Curve Loop(3) = {9, 10, 11, 8};
//+
Curve Loop(4) = {13, 14, 15, 12};
//+
Plane Surface(1) = {1,2,3,4};
//+
Extrude {0, 0, 0.15} {
  Surface{1}; 
}
//+
Physical Volume("volume") = {1};
//+
Physical Surface("triangle") = {55, 59, 51};
//+
Physical Surface("square") = {75, 63, 67, 71};
//+
Physical Surface("circle") = {79, 91, 83, 87};
//+
Physical Surface("sides") = {43, 39, 35, 47};
//+
Physical Surface("top") = {92};
//+
Physical Surface("bottom") = {1};
//+
Physical Curve("triangle") = {49, 50, 54};
//+
Physical Curve("square") = {61, 62, 66, 70};
//+
Physical Curve("circle") = {77, 82, 78, 86};
//+
Physical Curve("sides") = {38, 34, 33, 42};
//+
Physical Curve("triangle_c") = {23, 21, 22, 7, 5, 6};
//+
Physical Curve("square_c") = {24, 27, 26, 25, 8, 9, 10, 11};
//+
Physical Curve("circle_c") = {28, 31, 29, 30, 12, 13, 14, 15};
//+
Physical Curve("sides_c") = {18, 17, 20, 19, 4, 3, 2, 1};
//+
Physical Point("sides_c") = {18, 17, 4, 3, 26, 2, 22, 1};
//+
Physical Point("triangle_c") = {38, 7, 34, 6, 33, 5};
//+
Physical Point("square_c") = {45, 54, 10, 50, 9, 46, 8};
//+
Physical Point("circle_c") = {63, 61, 73, 68, 12, 16, 14, 15};
