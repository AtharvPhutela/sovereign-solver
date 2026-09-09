* Smoke instance -- primal infeasible.
* min   x
* s.t.  x >= 2   (C1)
*       x <= 1   (C2)
* No feasible point. Oracle must report status "infeasible", not a number.
NAME          TINY_INFEAS
ROWS
 N  COST
 G  C1
 L  C2
COLUMNS
    X         COST           1.0   C1             1.0
    X         C2             1.0
RHS
    RHS       C1             2.0   C2             1.0
ENDATA
