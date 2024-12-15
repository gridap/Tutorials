lc = 1e-1;

Point(1) = {0, -1.0000, 0, lc};
Point(2) = {0, 1.0000,  0, lc};
Point(3) = {1.9015, 1.6190, 0, lc};
Point(4) = {3.0777, 0, 0, lc};
Point(5) = {1.9015, -1.6190, 0, lc};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 1};

Curve Loop(1) = {1, 2, 3, 4, 5};

Plane Surface(1) = {1};

Physical Line("l1") = {1};
Physical Line("l2") = {2};
Physical Line("l3") = {3};
Physical Line("l4") = {4};
Physical Line("l5") = {5};

Physical Curve(“b1”) = {1, 2, 4};
Physical Curve("b2") = {3};

Physical Surface("My surface") = {1};

Mesh 2;
Save "pentagon_mesh.msh";
