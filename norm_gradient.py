import numpy as np
import sympy as sp

from scipy.optimize import golden


def steepest_gd(
        x_start: tuple[float, float],
        tolerance: float,
        max_iter: int = 100,
        lam_finder: str = "deriv"
) -> tuple[float, float]:
    x, y = sp.symbols("x y", real=True)
    lam = sp.Symbol("lambda", real=True)

    #function = x**2 + y**2
    function = x**2 - 4*x + y**2 - y - x*y

    gradient = sp.Matrix([
        sp.diff(function, x),
        sp.diff(function, y)
    ])

    x_curr, y_curr = map(float, x_start)

    for i in range(max_iter):
        print(f"\nIteration: {i + 1}")

        Xk = sp.Matrix([x_curr, y_curr])

        grad_Xk = gradient.subs({
            x: x_curr,
            y: y_curr
        })

        gradient_norm = np.linalg.norm(
            np.array(grad_Xk, dtype=float).reshape(-1)
        )

        print(f"Xk: {Xk.T}")
        print(f"Gradient: {grad_Xk.T}")
        print(f"Norma gradientu: {gradient_norm}")

        # Kontrola pred delením normou.
        if gradient_norm < tolerance:
            print("STOP: mala norma gradientu")
            break

        normalized_gradient = grad_Xk / gradient_norm

        Xk1 = Xk - lam * normalized_gradient

        phi_expr = sp.simplify(
            function.subs(
                {x: Xk1[0], y: Xk1[1]},
                simultaneous=True
            )
        )

        print(f"Normalizovany gradient: {normalized_gradient.T}")
        print(f"phi(lambda): {phi_expr}")

        if lam_finder == "deriv":
            d1 = sp.diff(phi_expr, lam)
            d2 = sp.diff(phi_expr, lam, 2)

            solutions = sp.solve(d1, lam)
            candidates = []

            for solution in solutions:
                if solution.free_symbols:
                    continue

                numeric = complex(sp.N(solution))

                if abs(numeric.imag) > 1e-10:
                    continue

                step = numeric.real

                if not np.isfinite(step) or step < 0:
                    continue

                curvature = float(d2.subs(lam, step)) # type: ignore

                if curvature > 0:
                    candidates.append(step)

            if not candidates:
                raise ValueError(
                    "Nenasiel sa vhodny krok cez deriváciu "
                    "Skss lam_finder='golden'"
                )

            lam_opt = min(
                candidates,
                key=lambda step: float(phi_expr.subs(lam, step))
            )

        else:
            phi = sp.lambdify(lam, phi_expr, "numpy")

            lam_opt = float(
                golden(phi, brack=(0.0, 1.0)) # type: ignore
            )

        X_new = Xk1.subs(lam, lam_opt)

        x_new = float(X_new[0])
        y_new = float(X_new[1])

        f_current = float(
            function.subs({x: x_curr, y: y_curr})
        )

        f_new = float(
            function.subs({x: x_new, y: y_new})
        )

        function_change = abs(f_new - f_current)

        position_change = np.linalg.norm([
            x_new - x_curr,
            y_new - y_curr
        ])

        print(f"Lambda: {lam_opt}")
        print(f"Novy bod: ({x_new}, {y_new})")
        print(f"f(novy bod): {f_new}")
        print(f"Zmena funkcie: {function_change}")
        print(f"Zmena polohy: {position_change}")

        x_curr, y_curr = x_new, y_new

        if position_change < tolerance:
            print("STOP: mala zmena polohy")
            break

        if function_change < tolerance:
            print("STOP: mala zmena funkcie")
            break

    print(f"\nVýsledný bod: ({x_curr}, {y_curr})")
    return x_curr, y_curr


if __name__ == "__main__":
    steepest_gd(
        x_start=(5.0, 5.0),
        tolerance=1e-6,
        max_iter=100,
        lam_finder="deriv"
    )
