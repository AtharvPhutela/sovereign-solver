* Smoke instance -- unbounded below.
* min  -x
* s.t.  x - y <= 5   (C1)
*       x, y >= 0
* x -> +inf with y = x - 5 stays feasible, objective -> -inf.
* Oracle must report status "unbounded", not a number.
NAME          TINY_UNBND
ROWS
 N  COST
 L  C1
COLUMNS
    X         COST          -1.0   C1             1.0
    Y         C1            -1.0
RHS
    RHS       C1             5.0
ENDATA
