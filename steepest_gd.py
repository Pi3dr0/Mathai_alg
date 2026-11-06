import math
import numpy as np
import sympy as sp


def steepest_gd(
        x_start: tuple[float, float],
        tolerance: float,
        max_iter: int = 100
) -> None:
    x, y, lam = sp.symbols("x, y, lambda")
    x_curr, y_curr = x_start

    function = 2*x**2 + x*y + 3*y**2 - 4*x + 2*y

    gradient = sp.Matrix([
        sp.diff(function, x),
        sp.diff(function, y)
    ])

    lam_solve = sp.symbols("lambda")

    for i in range(max_iter):
        print(f"Iteration: {i}")

        # step 1, 2 ========================
        Xk = sp.Matrix([x_curr, y_curr])
        print(f"X0: {Xk}")

        grad_Xk = gradient.subs({
            x: Xk[0],
            y: Xk[1]
        })
        print(f"grad_Xk: {grad_Xk}")

        Xk1 = Xk - lam * grad_Xk
        print(f"Xk1: {Xk1}")
        # ==================================
        
        # step 3 Dosadenie do f(x,y) =======
        Xk1p1 = function.subs({
            x: Xk1[0],
            y: Xk1[1]
        })
        print(f"f(d): {Xk1p1}")
        # ==================================

        # step 4 najdenie min f(d) =========
        d1 = sp.diff(Xk1p1, lam)
        lam_opt = sp.solve(sp.Eq(d1, 0), lam_solve)[0]

        d2 = sp.diff(Xk1p1, lam, 2)

        print(f"d1: {d1}")
        print(f"d2: {d2}")
        print(f"lambda opt: {lam_opt}")
        # ==================================

        # step 5 dosadime lam do Xk1p1 =====
        val = Xk1p1.subs({
            "lambda": lam_opt
        })
        print(val)
        

        # step 6 zastavovacie podmienky ===

        #! Work in progress!!

        # =================================

        # I2: X1 -> X2
        x_curr, y_curr = Xk - lam_opt * grad_Xk
        print(x_curr, y_curr)
        print("=================================")
    return



if __name__ == "__main__":
    max_iter: int = 10
    tolerance: float = 1e-6
    x_start: tuple[float, float] = (0.0, 0.0)

    steepest_gd(
        x_start=x_start,
        tolerance=tolerance,
        max_iter=max_iter
    )
