from util import *
from rbm import RestrictedBoltzmannMachine
import numpy as np
import matplotlib.pyplot as plt


def compute_hidden_probs(rbm, data, batch_size=100):
    """
    Compute p(h=1|v) for an entire dataset in mini-batches.
    Returns float32 array of shape (N, rbm.ndim_hidden).

    Using probabilities (not sampled activations) typically makes the next RBM
    training more stable because the input becomes less noisy.
    """
    N = data.shape[0]
    H = rbm.ndim_hidden
    out = np.zeros((N, H), dtype=np.float32)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        v = data[start:end, :]
        h_prob, h_act = rbm.get_h_given_v(v)
        out[start:end, :] = h_prob.astype(np.float32)

    return out


if __name__ == "__main__":

    image_size = [28, 28]
    train_imgs, train_lbls, test_imgs, test_lbls = read_mnist(dim=image_size, n_train=60000, n_test=10000)

    # --------------------
    # Settings match  Task 4.1 style
    # --------------------
    n_epochs = 10          # try 10 first; later 15 or 20
    batch_size = 20        # around 20 as specified
    nh1 = 500              # 784 -> 500
    nh2 = 500              # 500 -> 500

    # Optional reproducibility
    np.random.seed(0)

    print("\nTASK 4.2 (part 1): Greedy layer-wise pretraining of 2 RBMs (784-500-500)\n")

    # ============================================================
    # RBM 1: vis(784) -> hid(500)
    # ============================================================
    print("Training RBM1: 784 -> 500")

    rbm1 = RestrictedBoltzmannMachine(
        ndim_visible=image_size[0] * image_size[1],
        ndim_hidden=nh1,
        is_bottom=True,
        image_size=image_size,
        is_top=False,
        n_labels=10,
        batch_size=batch_size
    )

    # If you want RF images during training, keep rf["period"] small-ish.
    # If you want fewer files, increase this (or disable).
    # rbm1.rf["period"] = 5000

    hist1 = rbm1.cd1(
        visible_trainset=train_imgs,
        n_epochs=n_epochs,
        shuffle=True,
        return_history=True
    )

    rbm1_final_loss = hist1["recon_loss"][-1]
    print(f"RBM1 final avg recon loss = {rbm1_final_loss:.6f}")

    # ============================================================
    # Build dataset for RBM 2 by feeding forward through RBM 1
    # ============================================================
    print("\nComputing hidden representation for RBM2 training (p(h|v) from RBM1)...")
    train_h1 = compute_hidden_probs(rbm1, train_imgs, batch_size=200)  # use bigger batch for speed
    print(f"RBM2 training data shape: {train_h1.shape} (should be 60000 x 500)")

    # ============================================================
    # RBM 2: vis(500) -> hid(500)
    # ============================================================
    print("\nTraining RBM2: 500 -> 500")

    rbm2 = RestrictedBoltzmannMachine(
        ndim_visible=nh1,
        ndim_hidden=nh2,
        is_bottom=False,
        is_top=False,
        n_labels=10,
        batch_size=batch_size
    )

    # No receptive fields here 
    hist2 = rbm2.cd1(
        visible_trainset=train_h1,
        n_epochs=n_epochs,
        shuffle=True,
        return_history=True
    )

    rbm2_final_loss = hist2["recon_loss"][-1]
    print(f"RBM2 final avg recon loss = {rbm2_final_loss:.6f}")

    # ============================================================
    # Report + plots (what the lab asks you to report)
    # ============================================================
    print("\n--- Reconstruction loss report (avg per epoch) ---")
    print(f"RBM1 (784->500) final avg recon loss: {rbm1_final_loss:.6f}")
    print(f"RBM2 (500->500) final avg recon loss: {rbm2_final_loss:.6f}")

    # Plot recon loss curves
    plt.figure()
    plt.plot(hist1["recon_loss"], marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Avg reconstruction loss (MSE)")
    plt.title("RBM1 reconstruction loss per epoch (784 -> 500)")
    plt.show()

    plt.figure()
    plt.plot(hist2["recon_loss"], marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Avg reconstruction loss (MSE)")
    plt.title("RBM2 reconstruction loss per epoch (500 -> 500)")
    plt.show()