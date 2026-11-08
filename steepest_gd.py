import numpy as np
import sympy as sp

from scipy.optimize import golden


def steepest_gd(
        x_start: tuple[float, float],
        tolerance: float,
        max_iter: int = 100,
        lam_finder: str = "deriv"
) -> None:
    x, y, lam = sp.symbols("x, y, lambda")
    x_curr, y_curr = x_start
    x_bfr, y_bfr = 9999.9, 9999.9

    function = 2*x**2 + x*y + 3*y**2 - 4*x + 2*y

    gradient = sp.Matrix([
        sp.diff(function, x),
        sp.diff(function, y)
    ])


    for i in range(max_iter):
        print(f"Iteration: {i + 1}")

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

        # Najdenie lambdy podla derivace = 0
        if lam_finder == "deriv":
            d1 = sp.diff(Xk1p1, lam)
            lam_opt = sp.solve(sp.Eq(1, 0), lam)[0]
            d2 = sp.diff(Xk1p1, lam, 2)
            print(f"d1: {d1}")
            print(f"d2: {d2}")
        # najdenie lambdy podla Golden Search
        elif lam_finder == "golden":
            phi = sp.lambdify(lam, Xk1p1, "numpy")
            lam_opt = golden(phi, brack=(-5, 5))
        else:
            d1 = sp.diff(Xk1p1, lam)
            lam_opt = sp.solve(sp.Eq(1, 0), lam)[0]
            d2 = sp.diff(Xk1p1, lam, 2)
            print(f"d1: {d1}")
            print(f"d2: {d2}")

        print(f"lambda opt: {lam_opt}")
        # ==================================

        # step 5 dosadime lam do Xk1p1 =====

        val = Xk1p1.subs({
            "lambda": lam_opt
        })
        print(f"z = {val}") # z
        

        # step 6 zastavovacie podmienky ===
        f_Xk_val = function.subs({
            x: x_bfr,
            y: y_bfr
        })

        x_bfr, y_bfr = x_curr, y_curr

        f_Xkp1_val = function.subs({
            x: x_curr,
            y: y_curr
        })

        # Norma gradientu
        grad_Xk_num = np.array([
            float(grad_Xk[0]),
            float(grad_Xk[1])
        ])
        gradient_norm = np.linalg.norm(grad_Xk_num)

        # Zmena pozicie
        Xk1_num = Xk.subs(lam, lam_opt) - lam_opt * grad_Xk
        x_new = float(Xk1_num[0])
        y_new = float(Xk1_num[1])
        

         # I2: X1 -> X2
        print(f"X: {x_new} Y: {y_new}")

        # STOP: zmena hodnoty fynkcie
        print(f_Xkp1_val, f_Xk_val)
        change_f_val = abs(f_Xkp1_val - f_Xk_val)
        if change_f_val < tolerance: # STOP: Zmena hodnoty funkcie
            print(f"{change_f_val} < {tolerance} | Zmena hodnoty funkcie")
            break

        # STOP: norma gradientu
        if gradient_norm < tolerance: # STOP: Norma gradientu
            print(f"{gradient_norm} < {tolerance} | Norma gradientu")
            break

        # STOP: zmena polohy
        position_change = np.array([
            x_new - x_curr,
            y_new - y_curr
        ])
        position_norm = np.linalg.norm(position_change)
        if position_norm < tolerance: # STOP: Zmena polohy
            print(f"{position_norm} < {tolerance} | Zmena polohy")
            break

        #x_curr, y_curr = Xk - lam_opt * grad_Xk
        x_curr, y_curr = x_new, y_new
        print("=================================")
    return



if __name__ == "__main__":
    max_iter: int = 100
    tolerance: float = 0.000001
    x_start: tuple[float, float] = (0.0, 0.0)

    steepest_gd(
        x_start=x_start,
        tolerance=tolerance,
        max_iter=max_iter,
        lam_finder="golden"
    )
