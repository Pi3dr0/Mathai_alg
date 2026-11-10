import numpy as np
import sympy as sp

from scipy.optimize import golden


def steepest_gd(
    x_start: tuple[float, float],
    tolerance: float,
    max_iter: int = 100,
    lam_finder: str = "deriv"
) -> None:
    x, y, lam = sp.symbols("x y lambda")

    x_curr, y_curr = x_start

    # function = x**2 + y**2                                     # Sfera
    # function = x**2 + 25*y**2                                  # Pretiahnuta kvadraticka
    function = (x + 2*y - 7)**2 + (2*x + y - 5)**2               # Booth
    # function = 0.26*(x**2 + y**2) - 0.48*x*y                   # Matyas
    # function = (1 - x)**2 + 100*(y - x**2)**2                  # Rosenbrock
    # function = (x**2 + y - 11)**2 + (x + y**2 - 7)**2          # Himmelblau
    # function = (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2  # Beale
    # function = 2*x**2 + x*y + 3*y**2 - 4*x + 2*y

    gradient = sp.Matrix([
        sp.diff(function, x),
        sp.diff(function, y)
    ])

    for i in range(max_iter):
        print(f"Iteration: {i + 1}")

        # ==========================================================
        # Aktualny bod a gradient
        # ==========================================================
        Xk = sp.Matrix([x_curr, y_curr])
        print(f"Xk: {Xk}")

        grad_Xk = gradient.subs({
            x: x_curr,
            y: y_curr
        })
        print(f"grad_Xk: {grad_Xk}")

        grad_Xk_num = np.array([
            float(grad_Xk[0]),
            float(grad_Xk[1])
        ])

        gradient_norm = np.linalg.norm(grad_Xk_num)

        # ==========================================================
        # Symbolicky smer kroku: Xk1 = Xk + lambda * grad(Xk)
        # ==========================================================
        Xk1 = Xk + lam * grad_Xk
        print(f"Xk1: {Xk1}")

        # ==========================================================
        # Funkcia pozdlz smeru: f(Xk + lambda * grad(Xk))
        # ==========================================================
        Xk1p1 = function.subs({
            x: Xk1[0],
            y: Xk1[1]
        })
        print(f"f(lambda): {Xk1p1}")

        # ==========================================================
        # Najdenie optimalnej lambdy
        # ==========================================================
        if lam_finder == "deriv":
            d1 = sp.diff(Xk1p1, lam)
            d2 = sp.diff(Xk1p1, lam, 2)

            print(f"d1: {d1}")
            print(f"d2: {d2}")

            solutions = sp.solve(sp.Eq(d1, 0), lam)

            lam_opt = solutions[0]

        elif lam_finder == "golden":
            phi = sp.lambdify(lam, Xk1p1, "numpy")
            lam_opt = golden(phi, brack=(-5, 5))


        lam_opt = float(lam_opt) #type: ignore
        print(f"lambda opt: {lam_opt:+.10f}")

        # ==========================================================
        # Hodnota funkcie v optimalnej lambde
        # ==========================================================
        val = Xk1p1.subs(lam, lam_opt)
        print(f"z = {val}")

        # ==========================================================
        # Novy bod vypocitany numericky
        # ==========================================================
        x_new = x_curr + lam_opt * grad_Xk_num[0]
        y_new = y_curr + lam_opt * grad_Xk_num[1]

        print(f"X: {x_new} Y: {y_new}")

        # ==========================================================
        # Hodnoty funkcie pred a po kroku
        # ==========================================================
        f_curr = float(function.subs({
            x: x_curr,
            y: y_curr
        }))

        f_new = float(function.subs({
            x: x_new,
            y: y_new
        }))

        # ==========================================================
        # STOP: zmena hodnoty funkcie
        # ==========================================================
        change_f_val = abs(f_new - f_curr)

        if change_f_val < tolerance:
            print(f"{change_f_val} < {tolerance} | Zmena hodnoty funkcie")
            break

        # ==========================================================
        # STOP: norma gradientu
        # ==========================================================
        if gradient_norm < tolerance:
            print(f"{gradient_norm} < {tolerance} | Norma gradientu")
            break

        # ==========================================================
        # STOP: zmena polohy
        # ==========================================================
        position_change = np.array([
            x_new - x_curr,
            y_new - y_curr
        ])

        position_norm = np.linalg.norm(position_change)

        if position_norm < tolerance:
            print(f"{position_norm} < {tolerance} | Zmena polohy")
            break

        # Posun na novy bod
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
