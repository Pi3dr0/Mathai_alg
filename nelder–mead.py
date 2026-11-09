import random
import math


def nelderMead(
          points: list[list[float]],
          tolerance: float,
          alpha: float,
          gamma: float,
          rho: float,
          sigma: float,
          max_iter: int
          ) -> None:
     # Sort by value from function
     evaluated_points: list[tuple[float, list[float]]] = [
          (getVal(coordinates=point), point)
          for point in points
     ]
     evaluated_points.sort(
          key=lambda item: item[0]
     )

     for i in range(max_iter):
          evaluated_points.sort(
               key=lambda item: item[0]
          )

          distance_worst_best = math.hypot(
               evaluated_points[2][1][0] - evaluated_points[0][1][0],
               evaluated_points[2][1][1] - evaluated_points[0][1][1]
          )

          print(f"iteration: {i}")
          print(f"f val: {evaluated_points[2][0] - evaluated_points[0][0]}")
          print(f"Best point: {evaluated_points[0][1]}")

          # STOP: ak worst - best < tolerance
          if distance_worst_best < tolerance: break
          
          # Get center point between good and bad points
          center: tuple[float, list[float]] = getCenter(
               good_coor=evaluated_points[0][1],
               bad_coor=evaluated_points[1][1]
          )

          # Get reflex point
          reflection: tuple[float, list[float]] = getReflection(
               center_coor=center[1],
               worst_coor=evaluated_points[2][1],
               alpha=alpha
          )

          good_val: float = evaluated_points[0][0]
          bad_val: float = evaluated_points[1][0]

          # Compare good, bad and reflection
          if good_val <= reflection[0] < bad_val:
               evaluated_points[2] = reflection
               print("reflection 1")
               print("---------------------")
               continue

          # Check if reflection is the best val
          if reflection[0] < good_val:
               # Get expansion point
               expansion: tuple[float, list[float]] = getExpansion(
                    reflect_coor=reflection[1],
                    center_coor=center[1],
                    gamma=gamma
               )
               if reflection[0] < expansion[0]:
                    evaluated_points[2] = reflection
                    print("reflection 2")
                    print("---------------------")
                    continue
               else:
                    evaluated_points[2] = expansion
                    print("expansion")
                    print("---------------------")
                    continue

          worst_val: float = evaluated_points[2][0]
          
          # Check kontraction
          if reflection[0] >= bad_val:
               contraction: tuple[float, list[float]] = getContraction(
                    center_coor=center[1],
                    worst_coor=evaluated_points[2][1],
                    rho=rho
               )
               if contraction[0] < worst_val:
                    evaluated_points[2] = contraction
                    print("contraction")
                    print("---------------------")
                    continue
               else:
                    new_bad: tuple[float, list[float]] = getShrinking(
                         good_coor=evaluated_points[0][1],
                         point_coor=evaluated_points[1][1],
                         sigma=sigma
                    )
                    new_worst: tuple[float, list[float]] = getShrinking(
                         good_coor=evaluated_points[0][1],
                         point_coor=evaluated_points[2][1],
                         sigma=sigma
                    )
                    evaluated_points[1] = new_bad
                    evaluated_points[2] = new_worst
                    print("Shrinking")
                    print("---------------------")
                    continue
          print("---------------------")

     x_pos: float = evaluated_points[0][1][0]
     y_pos: float = evaluated_points[0][1][1]
     val: float = getVal(
          coordinates=[x_pos, y_pos]
     )     
     print(f"minimum: [{x_pos:.5f},{y_pos:.5f},{val:.5f}]")


def getCenter(
          good_coor: list[float],
          bad_coor: list[float]
     ) -> tuple[float, list[float]]:

     x_coor: float = (good_coor[0] + bad_coor[0]) / 2
     y_coor: float = (good_coor[1] + bad_coor[1]) / 2

     coordination: list[float] = [x_coor, y_coor]
     val: float = getVal(coordinates=coordination)

     return (val, coordination)


def getVal(coordinates: list[float]) -> float:
     val = function(
          x=coordinates[0],
          y=coordinates[1]
     )
     return val


def function(x: float, y:float) -> float:
     return x**2 + y**2 # Sfera
     #return x**2 + 25*y**2 # Pretiahnuta kvadraticka
     #return (x + 2*y - 7)**2 + (2*x + y - 5)**2 # Booth
     #return 0.26*(x**2 + y**2) - 0.48*x*y # Matyas
     #return (1-x)**2 + 100*(y-x**2)**2 # Rosenbrock
     #return (x**2 + y - 11)**2 + (x + y**2 - 7)**2 # Himmelblau
     #return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 -x + x*y**3)**2 # Beale
     #return  2*x**2 + x*y + 3*y**2 - 4*x + 2*y


def getReflection(
          center_coor: list[float],
          worst_coor: list[float],
          alpha: float
     ) -> tuple[float, list[float]]:
    
    x_coor: float = center_coor[0] + (alpha * (center_coor[0] - worst_coor[0]))
    y_coor: float = center_coor[1] + (alpha * (center_coor[1] - worst_coor[1]))

    coordination: list[float] = [x_coor, y_coor]
    val: float = getVal(coordinates=coordination)

    return (val, coordination)


def getContraction(
          center_coor: list[float],
          worst_coor: list[float],
          rho: float
     ) -> tuple[float, list[float]]:

     x_coor: float = center_coor[0] + (rho * (worst_coor[0] - center_coor[0]))
     y_coor: float = center_coor[1] + (rho * (worst_coor[1] - center_coor[1]))

     coordination: list[float] = [x_coor, y_coor]
     val: float = getVal(coordinates=coordination)

     return (val, coordination)


def getExpansion(
          reflect_coor: list[float],
          center_coor: list[float],
          gamma: float
          ) -> tuple[float, list[float]]:

     x_coor: float = (center_coor[0] + gamma * (reflect_coor[0] - center_coor[0]))
     y_coor: float = (center_coor[1] + gamma * (reflect_coor[1] - center_coor[1]))

     coordination: list[float] = [x_coor, y_coor]
     val: float = getVal(coordinates=coordination)

     return (val, coordination)


def getShrinking(
          good_coor: list[float],
          point_coor: list[float],
          sigma: float
     ) -> tuple[float, list[float]]:

     x_coor: float = good_coor[0] + (sigma * (point_coor[0] - good_coor[0]))
     y_coor: float = good_coor[1] + (sigma * (point_coor[1] - good_coor[1]))

     coordination: list[float] = [x_coor, y_coor]
     val: float = getVal(coordinates=coordination)

     return (val, coordination)


if __name__ == "__main__":
     points: list[list[float]] = [
          [3, 2], # sfera
          #[3.0, 1.0], # Pretiahnuta kvad.
          #[0.0, 0.0], # Booth
          #[5.0, -3.0], # Matyas
          #[-1.2, 1.0], # Rosebrock
          #[-4.0, 0.0], # Himmelblau
          #[1.0, 1.0], # Beale
          [1, 0], # *2
          [0, 1] # -3

     ]
     tolerance: float = 0.00001          # loop coefficient
     aplha: float = 1.0                 # reflection coefficient
     gamma: float = 2.0                 # expansion coefficient
     rho: float = 0.5                   # contraction coefficient
     sigma: float = 0.5                 # shrink coefficient
     max_iter: int = 20_000

     """for i in range(2):
          nd_point: float = points[i][0]
          rd_point: float = points[i][1]

          x = random.uniform(nd_point - 3, rd_point + 3)
          y = random.uniform(nd_point - 3, rd_point + 3)
          x = nd_point + 2
          y = rd_point - 3
          points.append([x, y])"""


     nelderMead(
          points=points,
          tolerance=tolerance,
          alpha=aplha,
          gamma=gamma,
          rho=rho,
          sigma=sigma,
          max_iter=max_iter
     )
