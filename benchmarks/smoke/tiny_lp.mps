* Smoke instance -- feasible bounded LP with a unique fractional optimum.
* min  -x - y
* s.t.  x + 2y <= 4      (C1)
*      4x + 2y <= 12     (C2)
*       x, y >= 0
* Optimum at the C1 n C2 intersection: x = 8/3, y = 2/3, objective = -10/3.
* Expected objective: -3.3333333333
NAME          TINY_LP
ROWS
 N  COST
 L  C1
 L  C2
COLUMNS
    X         COST          -1.0   C1             1.0
    X         C2             4.0
    Y         COST          -1.0   C1             2.0
    Y         C2             2.0
RHS
    RHS       C1             4.0   C2            12.0
BOUNDS
ENDATA
