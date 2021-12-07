import matplotlib.pyplot as plt


def visualize_train_valid_loss(train_valid_loss):
    """
    Plots training and validation losses.
    """
    plt.title("Train vs. Validation loss")
    plt.plot(train_valid_loss)
    plt.yscale("log")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend(["training loss", "validation loss"])
    plt.show()
