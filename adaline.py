def adaline(
        x_input: list[list[float]],
        y_input: list[float],
        lr: float,
        tolerance: float,
        max_iter: int
    ) -> None:
    number_of_x: int = len(x_input[0])
    number_of_rows: int = len(x_input)

    neuron_weight: list[float] = [
        0.0 for _ in range(number_of_x)
    ]
    bias: float = 1.0

    for epoch in range(max_iter):
        print("=======================")
        print(f"Epocha: {epoch+1}")
        max_weight_change: float = 0.0

        for i in range(number_of_rows):
            target: float = y_input[i]
            
            print("-----------------------")
            print(f"train data: {x_input[i]}")
            print(f"target: {target}")

            y_in: float = get_NET(
                train_data=x_input[i],
                neuron_weight=neuron_weight,
                bias=bias
            )

            error = target - y_in

            for x in x_input[i]:
                weight_change = abs(lr * error * x)
                max_weight_change = max(
                    max_weight_change,
                    weight_change
                )

            # weight updating
            neuron_weight = update_weights(
                train_data=x_input[i],
                neuron_weights=neuron_weight,
                lr=lr,
                error=error
            )
            bias += lr * error

            print(f"max weight change: {max_weight_change}")
            print(f"new: ({neuron_weight} | {bias:.4f})")

        if max_weight_change < tolerance:
            print("~~~~~~~~~~~~~~~~~~~~~~~")
            print(neuron_weight, bias)
            print(
                f"Output value: x1 {neuron_weight[0]:.3f}, x2 {neuron_weight[1]:.3f}, bias {bias:.3f}"
                )
            break



def get_NET( # dostanem y_in
        train_data: list[float],
        neuron_weight: list[float],
        bias: float
    ) -> float:
    return sum(x * w for x, w in zip(train_data, neuron_weight)) + bias


def update_weights(
        train_data: list[float],
        neuron_weights: list[float],
        lr: float,
        error: float
) -> list[float]:
    for i in range(len(neuron_weights)):
        neuron_weights[i] += lr * error * train_data[i]
    return neuron_weights



if __name__ == "__main__":
    lr: float = 0.001
    tolerance: float = 0.05
    max_iter: int = 500

    input_and: list[list[float]] = [
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0]
    ]
    output_and: list[float] = [
        1.0, -1.0, -1.0, -1.0
    ]


    test_input: list[list[float]] = [
            [-1, -1],
            [0, 0],
            [-1, 1],
            [2, -1],
            [1, 1],
    
            [2, 2],
            [1, 4],
            [3, 1],
            [0, 5],
            [5, 0]
        ]

    test_target: list[float] = [
            -1, -1, -1, -1, -1,
            1, 1, 1, 1, 1
        ]

    adaline(
        x_input=test_input,
        y_input=test_target,
        lr=lr,
        tolerance=tolerance,
        max_iter=max_iter
    )
