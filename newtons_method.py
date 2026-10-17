import math
import numpy as np
import sympy as sp
import random


# ================================================
# Pre 2D treba upravit premennu "f", pretoze inak
# to nebude fungovat s derivaciami z SymPy !
# ================================================


def function(x: float, y: float) -> float:
    #return math.exp(x) - 2 * x 
    #return (x+2)**2 + (y-1)**2 + (x*y)/2

    return x**2 + 4*x*y + y**2 - y - x*y


def newtonMethond1D(
        tolerance: float,
        x_k: float,
        max_iter: int
    ) -> None:
  
    x = sp.symbols('x')
    f = sp.exp(x) - 2*x

    # derivacie
    f_prime = sp.diff(f, x)
    f_double_prime = sp.diff(f, x, 2)

    x_new: float = x_k

    for i in range(max_iter):
        grad = float(f_prime.subs(x, x_k))              # type: ignore
        hess = float(f_double_prime.subs(x, x_k))       # type: ignore

        x_new: float = x_k - (grad/hess)

        print(f"iteration: [{i}]")
        print(f"x = {x_k:.5f}")
        print(f"x' = {grad:.5f}")
        print(f"x'' = {hess:.5f}")
        print(f"x_new = {x_new:.5f}")
        print("------------------------")

        stop_val: float = abs(x_new - x_k)
        if stop_val < tolerance: break

        x_k = x_new

    x_pos: float = x_new
    y_pos: float = function(x=x_pos, y=0)

    print(f"minimums: [{x_pos:.3f}, {y_pos:.3f}]")


def newtonMethond2D(
        x_start: float,
        y_start: float,
        tolerance: float,
        max_iter: int
    ) -> None:

    # Gradient
    x, y = sp.symbols("x, y")
    #f = (x+2)**2 + (y-1)**2 + (x*y)/2
    #f = x**2 + 4*x + y**2 - y - x*y

    f = x**2 + y**2 # Sfera
    #f = x**2 + 25*y**2 # Pretiahnuta kvadraticka
    #f = (x + 2*y - 7)**2 + (2*x + y - 5)**2 # Booth
    #f = 0.26*(x**2 + y**2) - 0.48*x*y # Matyas
    #f = (1-x)**2 + 100*(y-x**2)**2 # Rosenbrock
    #f = (x**2 + y - 11)**2 + (x + y**2 - 7)**2 # Himmelblau
    #f = (1.5 - x + x*y)**2 + (2.25 - x + x*y)**2 + (2.625 -x + x*y**3)**2 # Beale

    grad = sp.Matrix([
        sp.diff(f, x),
        sp.diff(f, y)
    ])

    # Hessian
    hess = sp.hessian(f, (x, y))
    print(hess)

    # Inverzna matica + SymPy -> NumPy
    """hess = np.array(hess.tolist(), dtype=float)
    hess_inv = np.linalg.inv(hess)
    x_k = np.array(
        [x_start, y_start],
        dtype=float
    )"""

    #! OVERIT CO JE TOTO...
    grad_func = sp.lambdify((x,y), grad, "numpy")
    hess_func = sp.lambdify((x,y), hess, "numpy")
    f_func = sp.lambdify((x,y), f, "numpy")

    x_k = np.array(
        [x_start, y_start],
        dtype=float
    )

    for i in range(max_iter):
        g = np.array(
            grad.subs({
                x: x_k[0],
                y: x_k[1]
            }).tolist(),
            dtype=float
        ).flatten()

        # == pridaie:
        Hk = np.array(hess_func(x_k[0], x_k[1]), dtype=float)

        hess_inv = np.linalg.inv(Hk)

        delta = -hess_inv @ g
        x_new = x_k + delta

        print(f"iteration: [{i+1}]")
        print(f"x = {x_k}")
        print(f"gradient = {g}")
        print(f"delta = {delta}")
        print(f"x_new = {x_new}")
        print("------------------------")

        if np.linalg.norm(g) < tolerance:
            break

        x_k = x_new


    val = f.subs({
        x: x_k[0],
        y: x_k[1]
    })
    
    print(f"minimum: [{x_k[0]:.3f}, {x_k[1]:.3f}, {val:.3f}]")
    

if __name__ == "__main__":
    tolerance: float = 0.001
    max_iter: int = 20000

    #stratring_point: list[float] = [3.0, 2.0] # Sfera
    #stratring_point: list[float] = [3.0, 1.0] # Pretiahnuta kvadraticka
    #stratring_point: list[float] = [0.0, 0.0] # Booth
    #stratring_point: list[float] = [5.0, -3.0] # Matyas
    #stratring_point: list[float] = [-1.2, 1] # Rosenbrock
    stratring_point: list[float] = [0.0, 0.0] # Himmelblau
    #stratring_point: list[float] = [1.0, 1.0] # Beale
    
    """stratring_point: list[float] = []
    for _ in range(2):
        x = random.uniform(-5, 5)
        y = random.uniform(-5, 5)
        stratring_point.append(x)
        stratring_point.append(y)"""
    

    """newtonMethond1D(
        tolerance=tolerance,
        x_k=stratring_point[0],
        max_iter=max_iter
    )"""

    newtonMethond2D(
        x_start=stratring_point[0],
        y_start=stratring_point[1],
        tolerance=tolerance,
        max_iter=max_iter
    )
