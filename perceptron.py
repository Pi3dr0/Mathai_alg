
def perceptron(
        train_data: list[list[int]],
        target: list[int],
        lr: float,
        threshold: float) -> None:
    number_of_x: int = len(train_data[0])
    number_of_rows: int = len(train_data)
    neuron_weight: list[float] = [0.0 for _ in range(number_of_x)]
    bias: float = 0
    max_epochs: int = 100

    # Iteracie while != 2
    for epoch in range(max_epochs):
        updated: bool = False

        # For row
        for i in range(number_of_rows):
            z_val: float = get_NET(
                train_data=train_data[i],
                neuron_weight=neuron_weight,
                bias=bias,
                reps=number_of_x
            )

            #! Toto opravit
            # +1 ak z > 0.2 threshold
            # 0 ak -0.2 <= z <= 0.2
            # -1 ak z < -0.2
            prediction: int = 1 if z_val >= threshold else -1

            if prediction != target[i]:
                error: int = target[i] - prediction
                neuron_weight = weight_diff(
                    lr=lr,
                    error=error,
                    train_data=train_data[i],
                    old_weights=neuron_weight
                )
                bias += lr * error
                updated = True

            print(f"epoch {epoch}: weights={neuron_weight}, bias={bias}")

            if not updated:
                print("Perceptro Konvergoval")
                break

    print(neuron_weight, bias)


def get_NET(train_data: list[int], neuron_weight: list, bias: float, reps: int) -> float:
    z: int = 0
    for i in range(reps):
        z += train_data[i] * neuron_weight[i]
    return z + bias


def weight_diff(
        lr: float,
        error: int,
        train_data: list[int],
        old_weights: list[float],
) -> list[float]:
    return [
        w + lr * error * x
        for w, x in zip(old_weights, train_data)
    ]




if __name__ == "__main__":
    lr: float = 1.0
    threshold: float = 0.2

    and_input: list[list[int]] = [
        [1, 0],
        [0, 1],
        [1, 0],
        [0, 0]
    ]
    and_target: list[int] = [1, -1, -1, -1]

    perceptron(
        train_data=and_input,
        target=and_target,
        lr=lr,
        threshold=threshold
    )

