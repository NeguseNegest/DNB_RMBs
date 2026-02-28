from util import *
from rbm import RestrictedBoltzmannMachine
import matplotlib.pyplot as plt
import numpy as np


def plot_reconstructions(rbm, test_imgs, n=20, seed=0, save_as=None):
    """
    Plots originals (top row) and reconstructions (bottom row).
    If save_as is provided, saves the figure to that filename.
    """
    rng = np.random.RandomState(seed)
    idx = rng.choice(test_imgs.shape[0], size=n, replace=False)
    v0 = test_imgs[idx, :]

    # One-step reconstruction: v0 -> h -> v1
    h_prob, h_act = rbm.get_h_given_v(v0)
    v1_prob, v1_act = rbm.get_v_given_h(h_act)

    fig, axs = plt.subplots(2, n, figsize=(n * 0.9, 2.2))
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.02, hspace=0.02)

    for i in range(n):
        axs[0, i].imshow(v0[i].reshape(28, 28), cmap="gray", vmin=0, vmax=1, interpolation=None)
        axs[0, i].set_xticks([])
        axs[0, i].set_yticks([])

        axs[1, i].imshow(v1_prob[i].reshape(28, 28), cmap="gray", vmin=0, vmax=1, interpolation=None)
        axs[1, i].set_xticks([])
        axs[1, i].set_yticks([])

    axs[0, 0].set_ylabel("orig", rotation=0, labelpad=20, va="center")
    axs[1, 0].set_ylabel("recon", rotation=0, labelpad=20, va="center")

    if save_as is not None:
        plt.savefig(save_as, dpi=200)
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":

    image_size = [28, 28]
    train_imgs, train_lbls, test_imgs, test_lbls = read_mnist(dim=image_size, n_train=60000, n_test=10000)

    # ---- Task 4.1 experimental protocol ----
    n_epochs = 10
    batch_size = 20

    hidden_units_list = [500, 400, 300, 200]

    SAVE_RF_FOR_HIDDEN = 500

    results_final_loss = {}
    results_history = {}
    trained_rbms = {}  # store trained rbm per hidden size so we can visualize later

    print("\nTASK 4.1: RBM CD1 training (epochs + minibatches) and hidden-units sweep\n")

    for nh in hidden_units_list:

        print(f"\n--- RBM: hidden={nh}, batch={batch_size}, epochs={n_epochs} ---")

        rbm = RestrictedBoltzmannMachine(
            ndim_visible=image_size[0] * image_size[1],
            ndim_hidden=nh,
            is_bottom=True,
            image_size=image_size,
            is_top=False,
            n_labels=10,
            batch_size=batch_size
        )

        # Avoid generating tons of rf pngs for every hidden size
        if nh != SAVE_RF_FOR_HIDDEN:
            rbm.rf["period"] = 10**12  # effectively "off"

        history = rbm.cd1(visible_trainset=train_imgs, n_epochs=n_epochs, shuffle=True, return_history=True)

        results_history[nh] = history
        results_final_loss[nh] = history["recon_loss"][-1]
        trained_rbms[nh] = rbm

        print(f"Final avg recon loss (hidden={nh}) = {results_final_loss[nh]:.6f}")

    # ---- Plot: final avg reconstruction loss vs hidden units ----
    xs = list(hidden_units_list)
    ys = [results_final_loss[h] for h in xs]

    plt.figure()
    plt.plot(xs, ys, marker='o')
    plt.xlabel("Number of hidden units")
    plt.ylabel("Final avg reconstruction loss (MSE)")
    plt.title(f"RBM: recon loss vs hidden units (epochs={n_epochs}, batch={batch_size})")
    plt.gca().invert_xaxis()
    plt.show()

    # ---- Optional: stability plots (recon loss, ||dW||, hidden activity) ----
    plt.figure()
    for h in hidden_units_list:
        plt.plot(results_history[h]["recon_loss"], label=f"h={h}")
    plt.xlabel("Epoch")
    plt.ylabel("Avg reconstruction loss (MSE)")
    plt.title("Stability: reconstruction loss per epoch")
    plt.legend()
    plt.show()

    plt.figure()
    for h in hidden_units_list:
        plt.plot(results_history[h]["mean_dW_norm"], label=f"h={h}")
    plt.xlabel("Epoch")
    plt.ylabel("Avg ||ΔW|| per epoch")
    plt.title("Stability: parameter update norm per epoch")
    plt.legend()
    plt.show()

    plt.figure()
    for h in hidden_units_list:
        plt.plot(results_history[h]["mean_h_prob"], label=f"h={h}")
    plt.xlabel("Epoch")
    plt.ylabel("Avg hidden ON-probability")
    plt.title("Stability: hidden activity per epoch")
    plt.legend()
    plt.show()

 
    for nh in hidden_units_list:
        plot_reconstructions(trained_rbms[nh], test_imgs, n=20, seed=0, save_as=f"recon.hidden{nh}.png")
