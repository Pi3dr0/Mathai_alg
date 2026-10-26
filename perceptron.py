def perceptron(
    train_data: list[list[int]],
    target: list[int],
    lr: float,
    threshold: float,
    max_iter: int,
) -> None:
    number_of_x: int = len(train_data[0])
    number_of_rows: int = len(train_data)

    neuron_weight: list[float] = [
        0.0 for _ in range(number_of_x)
    ]
    bias: float = 0.0

    for epoch in range(max_iter):
        updated: bool = False

        print("=====================")
        print(f"Epoch: {epoch + 1}")

        for i in range(number_of_rows):
            print("-----------------------")
            print(f"input data: {train_data[i]}")

            z_val: float = get_NET(
                train_data=train_data[i],
                neuron_weight=neuron_weight,
                bias=bias,
                reps=number_of_x,
            )
            print(f"NET: {z_val} | target: {target[i]}")

            if z_val > threshold:
                prediction = 1
            elif z_val < -threshold:
                prediction = -1
            else:
                prediction = 0

            print(f"aktivacna funkcia: {prediction}")

            if prediction != target[i]:
                weight_change: list[float] = weight_diff(
                    lr=lr,
                    target=target[i],
                    train_data=train_data[i],
                )
                bias_change: float = lr * target[i]

                for j in range(number_of_x):
                    neuron_weight[j] += weight_change[j]

                bias += bias_change
                updated = True

                print(
                    f"weight change: {weight_change} | "
                    f"bias change: {bias_change}"
                )

            print(
                f"new weights: {neuron_weight} | "
                f"bias: {bias}"
            )

        if not updated:
            print("=====================")
            print(
                f"Trenovanie skoncilo v epoche {epoch + 1}: "
                "ziadna aktualizacia."
            )
            break
    else:
        print("=====================")
        print(
            f"Dosiahnuty maximalny pocet epoch ({max_iter}) "
            "bez epochy bez aktualizacii."
        )

    print(f"Final weights: {neuron_weight} | bias: {bias}")


def get_NET(
    train_data: list[int],
    neuron_weight: list[float],
    bias: float,
    reps: int,
) -> float:
    z: float = 0.0

    for i in range(reps):
        z += train_data[i] * neuron_weight[i]

    return z + bias


def weight_diff(
    lr: float,
    target: int,
    train_data: list[int],
) -> list[float]:
    return [
        lr * target * x
        for x in train_data
    ]


if __name__ == "__main__":
    lr: float = 1.0
    threshold: float = 0.2
    max_iter: int = 10

    and_input: list[list[int]] = [
        [1, 1],
        [1, 0],
        [0, 1],
        [0, 0],
    ]
    and_target: list[int] = [1, -1, -1, -1]

    perceptron(
        train_data=and_input,
        target=and_target,
        lr=lr,
        threshold=threshold,
        max_iter=max_iter,
    )