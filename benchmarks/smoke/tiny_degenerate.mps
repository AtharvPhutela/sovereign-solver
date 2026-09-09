* Smoke instance -- primal-degenerate optimum.
* min  -x - y
* s.t.  x + y <= 2   (C1)
*       x     <= 1   (C2)
*           y <= 1   (C3)
*       x, y >= 0
* Optimum at (1, 1) with objective -2. All three constraints are active there,
* so any optimal basis carries a zero-valued basic variable -- the case an
* anti-cycling rule has to survive (Build Map #4).
NAME          TINY_DEGEN
ROWS
 N  COST
 L  C1
 L  C2
 L  C3
COLUMNS
    X         COST          -1.0   C1             1.0
    X         C2             1.0
    Y         COST          -1.0   C1             1.0
    Y         C3             1.0
RHS
    RHS       C1             2.0   C2             1.0
    RHS       C3             1.0
ENDATA
