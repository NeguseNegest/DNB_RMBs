from util import *
from dbn import DeepBeliefNet
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    image_size = [28, 28]
    train_imgs, train_lbls, test_imgs, test_lbls = read_mnist(dim=image_size, n_train=60000, n_test=10000)

    # DBN architecture: 784-500-500-(500+10)-2000
    dbn = DeepBeliefNet(
        sizes={"vis": 784, "hid": 500, "pen": 500, "top": 2000, "lbl": 10},
        image_size=image_size,
        n_labels=10,
        batch_size=20
    )

    # Train greedily + top RBM with labels
    # - n_epochs is recommended so you can report recon loss per epoch easily (10..20)
    # - n_iterations is still there for compatibility, but n_epochs controls training here
    n_epochs = 10
    dbn.train_greedylayerwise(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=10000, n_epochs=n_epochs)

    print("\nRecognition on train set (may take time)...")
    _, train_maxprob, train_acc_curve = dbn.recognize(train_imgs[:10000], train_lbls[:10000], return_convergence=True)

    print("\nRecognition on test set...")
    _, test_maxprob, test_acc_curve = dbn.recognize(test_imgs, test_lbls, return_convergence=True)

    # Plot label convergence (mean max label probability vs Gibbs iteration)
    plt.figure()
    plt.plot(test_maxprob, marker='o')
    plt.xlabel("Gibbs iteration")
    plt.ylabel("Mean max label probability")
    plt.title("Label convergence (test): mean max prob vs Gibbs iteration")
    plt.show()

    # Plot accuracy vs Gibbs iteration 
    plt.figure()
    plt.plot(test_acc_curve * 100.0, marker='o')
    plt.xlabel("Gibbs iteration")
    plt.ylabel("Accuracy (%)")
    plt.title("Recognition accuracy vs Gibbs iteration (test)")
    plt.show()