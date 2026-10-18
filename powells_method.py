import math


def goldenSearch(
        a: float,
        b: float,
        x_curr: float,
        y_curr: float,
        u: tuple[float, float],
        tolerance: float
    ) -> float:

    tho: float = (1 + math.sqrt(5))/2

    while b - a > tolerance:
        c: float = b - (b - a) / tho
        d: float = a + (b - a) / tho

        gc = lineFunction(
            t=c,
            x_curr=x_curr,
            y_curr=y_curr,
            u=u
        )
        gd = lineFunction(
            t=d,
            x_curr=x_curr,
            y_curr=y_curr,
            u=u
        )

        if gc < gd: b = d
        else: a = c

    return (a + b) / 2


def g_tx(t: float, x_curr: float, y_curr: float) -> float:
    return (x_curr + t)**2 + y_curr**2 # sfera
    #return (x_curr + t)**2 + 25*y_curr**2
    """return (
        ((x_curr + t) + 2*y_curr - 7)**2
        + (2*(x_curr + t) + y_curr - 5)**2
    )""" # Booth
    """return (
        0.26*((x_curr + t)**2 + y_curr**2)
        - 0.48*(x_curr + t)*y_curr
    )""" # Matyas
    """return (
        (1 - (x_curr + t))**2
        + 100*(y_curr - (x_curr + t)**2)**2
    )""" # Rosenbrock
    """return (
        ((x_curr + t)**2 + y_curr - 11)**2
        + ((x_curr + t) + y_curr**2 - 7)**2
    )""" # Himmelblau
    """return (
        (1.5 - (x_curr + t) + (x_curr + t)*y_curr)**2
        + (2.25 - (x_curr + t) + (x_curr + t)*y_curr)**2
        + (2.625 - (x_curr + t) + (x_curr + t)*y_curr**3)**2
    )""" # Beale


def g_ty(t: float, x_curr: float, y_curr: float) -> float:
    return x_curr**2 + (y_curr + t)**2 # sfera
    #return x_curr**2 + 25*(y_curr + t)**2
    """return (
        (x_curr + 2*(y_curr + t) - 7)**2
        + (2*x_curr + (y_curr + t) - 5)**2
    )""" # Booth
    """return (
        0.26*(x_curr**2 + (y_curr + t)**2)
        - 0.48*x_curr*(y_curr + t)
    )""" # Matyas
    """return (
        (1 - x_curr)**2
        + 100*((y_curr + t) - x_curr**2)**2
    )""" # Rosenbrock
    """return (
        (x_curr**2 + (y_curr + t) - 11)**2
        + (x_curr + (y_curr + t)**2 - 7)**2
    )""" # Himmelblau
    """return (
        (1.5 - x_curr + x_curr*(y_curr + t))**2
        + (2.25 - x_curr + x_curr*(y_curr + t))**2
        + (2.625 - x_curr + x_curr*(y_curr + t)**3)**2
    )""" # Beale


def function(x: float, y: float) -> float:
    return x**2 + y**2 # sfera
    #return x**2 + 25*y**2 # Pretiahnuta kvadraticka
    #return (x + 2*y - 7)**2 + (2*x + y - 5)**2 # Booth
    #return 0.26*(x**2 + y**2) - 0.48*x*y # Matyas
    #return (1-x)**2 + 100*(y-x**2)**2 # Rosenbrock
    #return (x**2 + y - 11)**2 + (x + y**2 - 7)**2 # Himmelblau
    #return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 -x + x*y**3)**2 # Beale


def lineFunction(
        t: float,
        x_curr: float,
        y_curr: float,
        u: tuple[float, float]
    ) -> float:
    x_new: float = x_curr + t * u[0]
    y_new: float = y_curr + t * u[1]
    return function(x=x_new, y=y_new)


def lineSearch(
        x_curr: float,
        y_curr: float,
        u: tuple[float, float],
        a: float,
        b: float,
        tolerance: float
    ) -> tuple[float, float]:
    t_start = goldenSearch(
        a=a,
        b=b,
        x_curr=x_curr,
        y_curr=y_curr,
        u=u,
        tolerance=tolerance
    )

    x_new = x_curr + t_start * u[0]
    y_new = y_curr + t_start * u[1]

    return x_new, y_new


def powellsMethod(
        x_start: float,
        y_start: float,
        max_iter: int,
        tolerance: float,
        x_end: float,
        y_end: float
    ) -> None:
    x_curr, y_curr = x_start, y_start
    u1: tuple[float, float] = (1.0, 0.0) # na osi X
    u2: tuple[float, float] = (0.0, 1.0) # na osi Y
    a, b = -10.0, 10.0


    for i in range(max_iter):

        # ==========================================
        # ukoncenie podla zadania z cvika
        distance_to_min: float = math.sqrt(
            (x_curr-x_end)**2 + (y_curr-y_end)**2
        )
        if distance_to_min < tolerance: break
        # ==========================================

        x_iter_start, y_iter_start = x_curr, y_curr

        # Finding in first direction
        x_curr, y_curr = lineSearch(
            x_curr=x_curr,
            y_curr=y_curr,
            u=u1,
            a=a,
            b=b,
            tolerance=tolerance
        )

        # Finding in second direction
        x_curr, y_curr = lineSearch(
            x_curr=x_curr,
            y_curr=y_curr,
            u=u2,
            a=a,
            b=b,
            tolerance=tolerance
        )

        # New direction
        v: tuple[float, float] = (
            x_curr - x_iter_start,
            y_curr - y_iter_start
        )
        if v != (0.0, 0.0):
            # Searching in new point
            x_curr, y_curr = lineSearch(
                x_curr=x_curr,
                y_curr=y_curr,
                u=v,
                a=a,
                b=b,
                tolerance=tolerance
            )

            u1, u2 = u2, v

        print(f"Iteration: {i+1}")
        print(f"x start: {x_iter_start:.5f} | x_new: {x_curr:.5f}")
        print(f"y start: {y_iter_start:.5f} | y_new: {y_curr:.5f}")
        print(f"new direction v: [{v[0]:.5f}. {v[1]:.5f}]")
        print(f"Way for new cycle: [{u1[0]:.5f}, {u1[1]:.5f}], [{u2[0]:.5f}, {u2[1]:.5f}]")
        print("--------------------------------------------------------------")

    val: float = function(x=x_curr, y=y_curr)
    print(f"minimum: [{x_curr:.3f}, {y_curr:.3f}, {val:.3f}]")
    return


if __name__ == "__main__":
    max_iter: int = 20_000
    tolerance: float = 1e-3 # 0.001

    starting_point: tuple[float, float] = (3.0, 2.0) # sfera
    #starting_point: list[float] = [3.0, 1.0] # Pretiahnuta kvadraticka
    #starting_point: list[float] = [0.0, 0.0] # Booth
    #starting_point: list[float] = [5.0, -3.0] # Matyas
    #starting_point: list[float] = [-1.2, 1] # Rosenbrock
    #starting_point: list[float] = [0.0, 0.0] # Himmelblau
    #starting_point: list[float] = [1.0, 1.0] # Beale

    end_point: tuple[float, float] = (0.0, 0.0) # sfera
    #end_point: tuple[float, float] = (0.0, 0.0) # Pretiahnuta kvadraticka
    #end_point: tuple[float, float] = (1.0, 3.0) # Booth
    #end_point: tuple[float, float] = (0.0, 0.0) # Matyas
    #end_point: tuple[float, float] = (1.0, 1.0) # Rosenbrock
    #end_point: tuple[float, float] = (3.0, 2.0) # Himmelblau
    #end_point: tuple[float, float] = (3.0, 0.5) # Beale


    powellsMethod(
        x_start=starting_point[0],
        y_start=starting_point[1],
        max_iter=max_iter,
        tolerance=tolerance,
        x_end=end_point[0],
        y_end=end_point[1]
    )
