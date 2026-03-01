from util import *
from dbn import DeepBeliefNet
import numpy as np
import os


if __name__ == "__main__":
    image_size = [28, 28]
    train_imgs, train_lbls, test_imgs, test_lbls = read_mnist(dim=image_size, n_train=60000, n_test=10000)

    np.random.seed(0)

    dbn = DeepBeliefNet(
        sizes={"vis": 784, "hid": 500, "pen": 500, "top": 2000, "lbl": 10},
        image_size=image_size,
        n_labels=10,
        batch_size=20,
    )

    # At least 200 as required by the assignment text
    dbn.n_gibbs_gener = 200

    # Greedy pretraining (loads from trained_rbm/ if present)
    n_epochs = 10
    dbn.train_greedylayerwise(
        vis_trainset=train_imgs,
        lbl_trainset=train_lbls,
        n_iterations=10000,
        n_epochs=n_epochs,
    )

    # Generate multiple independent chains per digit by changing the output "name"
    n_chains_per_digit = 3
    out_dir = "generated_mp4"
    os.makedirs(out_dir, exist_ok=True)

    base_name = os.path.join(out_dir, "task4.2")

    for digit in range(10):
        digit_1hot = np.zeros((1, 10), dtype=np.float32)
        digit_1hot[0, digit] = 1.0

        for chain in range(n_chains_per_digit):
            name = f"{base_name}.chain{chain}"
            dbn.generate(digit_1hot, name=name)
