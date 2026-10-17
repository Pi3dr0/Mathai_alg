import math
import numpy as np
import sympy as sp # derivacie atd...
import random 


def gradDesc2D(
        alpha: float,
        tolerance: float,
        x_start: float,
        y_start: float,
        x_end: float,
        y_end: float,
        max_iter: int = 20_000
    ) -> None:
    x, y = sp.symbols("x, y")
    function_val:float = 0.0
    x_curr, y_curr = x_start, y_start

    #function = x**2 + y**2 # Sfera
    #function = x**2 + 25*y**2 # Pretiahnuta kvadraticka
    #function = (x + 2*y - 7)**2 + (2*x + y - 5)**2 # Booth
    #function = 0.26*(x**2 + y**2) - 0.48*x*y # Matyas
    #function = (1-x)**2 + 100*(y-x**2)**2 # Rosenbrock
    function = (x**2 + y - 11)**2 + (x + y**2 - 7)**2 # Himmelblau
    #function = (1.5 - x + x*y)**2 + (2.25 - x + x*y)**2 + (2.625 -x + x*y**3)**2 # Beale


     # Gradient of function
    grad = sp.Matrix([
        sp.diff(function, x),
        sp.diff(function, y)
    ])
    grad_f = sp.lambdify(
        (x, y),
        grad,
        "numpy"
    )
    funcction_f = sp.lambdify(
        (x, y),
        function,
        "numpy"
    )

    # Loop
    for i in range(max_iter):
        g = grad_f(x_curr, y_curr)

        gx: float = g[0, 0]
        gy: float = g[1, 0]

        # =============================
        # podla alg.
        """abs_grad: float = math.sqrt(
            gx**2 + gy**2
        )

        if abs_grad < tolerance: break"""
        # =============================

        # =============================
        # podla zadania
        distance_to_min = math.hypot(
            x_curr - x_end,
            y_curr - y_end
        )
        if distance_to_min < tolerance: break
        # =============================

        x_curr -= alpha * gx
        y_curr -= alpha * gy

        function_val: float = funcction_f(
            x_curr,
            y_curr
        )

        print(f"Iteration: [{i+1}]")
        print(f"new x: {x_curr:.5f}")
        print(f"new y: {y_curr:.5f}")
        print(f"func. val: {function_val:.5f}")
        print("------------------------")

    print(f"minimum: [{x_curr:.3f}, {y_curr:.3f}, {function_val:.3f}]")
    return


if __name__ == "__main__":
    lr: float = 0.02
    max_iter: int = 20_000
    tolerance: float = 0.001

    #starting_points: tuple[float, float] = (3.0, 2.0) # sfera
    #starting_points: tuple[float, float] = (3.0, 1.0) # Pretiahnuta kvadraticka
    #starting_points: tuple[float, float] = (0.0, 0.0) # Booth
    #starting_points: tuple[float, float] = (5.0, -3.0) # Matyas
    #starting_points: tuple[float, float] = (-1.2, 1.0) # Rosenbrock
    starting_points: tuple[float, float] = (0.0, 0.0) # Himmelblau
    #starting_points: tuple[float, float] = (1.0, 1.0) # Beale

    #end_poit: tuple[float, float] = (0.0, 0.0) # sfera
    #end_poit: tuple[float, float] = (0.0, 0.0) # Pretiahnuta kvadraticka
    #end_poit: tuple[float, float] = (1.0, 3.0) # Booth
    #end_poit: tuple[float, float] = (0.0, 0.0) # Matyas
    #end_poit: tuple[float, float] = (1.0, 1.0) # Rosenbrock
    end_poit: tuple[float, float] = (3.0, 2.0) # Himmelblau
    #end_poit: tuple[float, float] = (3.0, 0.5) # Beale


    gradDesc2D(
        alpha=lr,
        max_iter=max_iter,
        tolerance=tolerance,
        x_start=starting_points[0],
        y_start=starting_points[1],
        x_end=end_poit[0],
        y_end=end_poit[1]
    )
