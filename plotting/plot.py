import matplotlib.pyplot as plt


# TODO: edit path name - refactor, put in different folder - utils?
# Todo type hinting
def make_plots(train_loss, train_accuracy, val_loss, val_accuracy, path_name) -> None:

    # with open('C:/path/numbers.txt') as f:
    # lines = f.read().splitlines()
    plt.figure(figsize=(10, 7))
    plt.plot(train_accuracy, color="green", label="train accuracy")
    plt.plot(val_accuracy, color="blue", label="validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(config["output_path"], path_name))

    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color="orange", label="train loss")
    plt.plot(val_loss, color="red", label="validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(config["output_path"], path_name))
