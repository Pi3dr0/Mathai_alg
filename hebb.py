
def hebb(
        input_data: list[list[float]],
        input_target: list[float],
        lr: float,
        max_iter: int
) -> None:
    num_of_x: int = len(input_data[0])
    num_of_rows: int = len(input_data)

    neurons_weight: list[float] = [
        0.0 for _ in range(num_of_x)
    ]
    bias: float = 0.0

    for epoch in range(max_iter):
        print("=====================")
        print(f"Epocha: {epoch+1}")

        for i in range(num_of_rows):
            print("-----------------------")
            print(f"train data: {input_data[i]}")

            z_val: float = get_NET(
                train_data=input_data[i],
                neurons_weight=neurons_weight,
                bias=bias,
                reps=num_of_x
            )
            print(f"NET: {z_val}")
            print(f"target: {input_target[i]}")

            # weight diff
            weight_change = weght_diff(
                lr=lr,
                target=input_target[i],
                train_data=input_data[i]
            )
            bias_change = lr * input_target[i]

            # update
            for j in range(num_of_x):
                neurons_weight[j] += weight_change[j]
            bias += bias_change

        # overenie
        done_training: bool = False
        for i in range(num_of_rows):
            prediction: float = get_NET(
                train_data=input_data[i],
                neurons_weight=neurons_weight,
                bias=bias,
                reps=num_of_x
            )
            if prediction >= 0:
                done_training = True
                break

        if done_training:
            break

    print(f"{neurons_weight}, {bias}")
    return


def get_NET(
        train_data: list[float],
        neurons_weight: list[float],
        bias: float,
        reps: int
) -> float:
    z: float = 0.0
    for i in range(reps):
        z += train_data[i] * neurons_weight[i]
    return z + bias


def weght_diff(
        lr: float,
        target: float,
        train_data: list[float]
) -> list[float]:
    return [
        lr * target * x
        for x in train_data
    ]



if __name__ == "__main__":
    lr: float = 1.0
    max_iter: int = 100

    and_input: list[list[float]] = [
        [1, 1],
        [1, -1],
        [-1, 1],
        [-1, -1]
    ]

    and_target: list[float] = [1, -1, -1, -1]

    hebb(
        input_data=and_input,
        input_target=and_target,
        lr=lr,
        max_iter=max_iter
    )
