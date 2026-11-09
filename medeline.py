import numpy as np


def madaline(
        train_data,
        target_data: list[float],
        lr: float,
        threshold: float,
        max_epoch: int
) -> None:
    w11, w12 = 0.0, 0.0
    w21, w22 = 0.0, 0.0
    b1, b2 = 0.0, 0.0

    v1, v2 = 0.5, 0.5
    b3 = 0.5

    for epoch in range(max_epoch):
        errors = 0

        for inputs, target in zip(train_data, target_data):
            x1, x2 = inputs
            print("~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"x1: {x1}, x2: {x2}")

            z1_in = get_NET(
                x1=x1,
                x2=x2,
                w1=w11,
                w2=w21,
                b=b1
            )

            z2_in = get_NET(
                x1=x1,
                x2=x2,
                w1=w12,
                w2=w22,
                b=b2
            )
            

            z1 = activation_function(zn_in=z1_in, threshold=threshold)
            z2 = activation_function(zn_in=z2_in, threshold=threshold)

            print(f"z1_in: {z1_in} | z1: {z1}")
            print(f"z2_in: {z2_in} | z1: {z2}")

            y_in = get_NET(
                x1=z1,
                x2=z2,
                w1=v1,
                w2=v2,
                b=b3
            )
            y = activation_function(zn_in=y_in, threshold=threshold)

            print(f"y_in: {y_in} | z1: {y} | target: {target}")

            if y == target:
                print(f"y == targrt | {y} == {target}")
                continue

            errors += 1

            if target == 1:
                print(f"{target} == 1")
                if abs(z1_in) <= abs(z2_in):
                    error = 1 - z1_in
                    w11_change = lr * error * x1
                    w21_change = lr * error * x2
                    b1_change = lr * error

                    print(f"Updating z1:")
                    print(f"error = 1 - {z1_in} = {error}")
                    print(f"w11 + w11_change: {w11} + {w11_change} = {w11 + w11_change}")
                    print(f"w21 + w21_change: {w21} + {w21_change} = {w21 + w21_change}")
                    print(f"bias1 + bias1_change: {b1} + {b1_change} = {b1 + b1_change}")

                    w11 += w11_change
                    w21 += w21_change
                    b1 += b1_change

                else:
                    error = 1 - z2_in
                    w12_change = lr * error * x1
                    w22_change = lr * error * x2
                    b2_change = lr * error

                    print(f"Updating z2:")
                    print(f"error = 1 - {z2_in} = {error}")
                    print(f"w12 + w12_change: {w12} + {w12_change} = {w12 + w12_change}")
                    print(f"w22 + w22_change: {w22} + {w22_change} = {w22 + w22_change}")
                    print(f"bias2 + bias2_change: {b2} + {b2_change} = {b2 + b2_change}")

                    w12 += w12_change
                    w22 += w22_change
                    b2 += b2_change
            else:
                print(f"{target} != 1")
                if z1 == 1:
                    error = -1 - z1_in
                    w11_change = lr * error * x1
                    w21_change = lr * error * x2
                    b1_change = lr * error

                    print(f"Updating z1:")
                    print(f"error = -1 - {z1_in} = {error}")
                    print(f"w11 + w11_change: {w11} + {w11_change} = {w11 + w11_change}")
                    print(f"w21 + w21_change: {w21} + {w21_change} = {w21 + w21_change}")
                    print(f"bias1 + bias1_change: {b1} + {b1_change} = {b1 + b1_change}")

                    w11 += w11_change
                    w21 += w21_change
                    b1 += b1_change
                if z2 == 1:
                    error = -1 - z2_in
                    w12_change = lr * error * x1
                    w22_change = lr * error * x2
                    b2_change = lr * error

                    print(f"Updating z2:")
                    print(f"error = -1 - {z2_in} = {error}")
                    print(f"w12 + w12_change: {w12} + {w12_change} = {w12 + w12_change}")
                    print(f"w22 + w22_change: {w22} + {w22_change} = {w22 + w22_change}")
                    print(f"bias2 + bias2_change: {b2} + {b2_change} = {b2 + b2_change}")

                    w12 += w12_change
                    w22 += w22_change
                    b2 += b2_change

                print("------------------------------")
        print(f"Epoch {epoch + 1}, error: {errors}")

        if errors == 0:
            break

    print(f"\nZ1: w11={w11:.4f}, w21={w21:.4f}, b1={b1:.4f}")
    print(f"Z2: w12={w12:.4f}, w22={w22:.4f}, b2={b2:.4f}")
        
    print("\nKontrola výsledkov:")

    for inputs, target in zip(train_data, target_data):
        x1, x2 = inputs

        z1_in = get_NET(x1, x2, w11, w21, b1)
        z2_in = get_NET(x1, x2, w12, w22, b2)

        z1 = activation_function(z1_in, threshold)
        z2 = activation_function(z2_in, threshold)

        y_in = get_NET(z1, z2, v1, v2, b3)
        y = activation_function(y_in, threshold)

        print(f"Vstup: {inputs}, cieľ: {target}, výstup: {y}")

    return


def get_NET(x1, x2, w1, w2, b) -> float:
    return (x1 * w1) + (x2 * w2) + b


def activation_function(zn_in, threshold) -> float:
    return 1 if zn_in >= threshold else -1



if __name__ == "__main__":
    lr: float = 0.2
    threshold: float = 0
    max_epoch: int = 10_000

    training_input = np.array([
        [1, 1],
        [1, -1],
        [-1, 1],
        [-1, -1]
    ], dtype=float)

    target_input: list[float] = [1, -1, -1, -1]

    madaline(
        train_data=training_input,
        target_data=target_input,
        lr=lr,
        threshold=threshold,
        max_epoch=max_epoch
    )
