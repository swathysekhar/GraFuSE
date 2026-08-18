"""GraFuSE project module.
"""

import matplotlib.pyplot as plt

def plot_training_history(history, save_path=None):
    plt.figure()
    plt.plot(history.get("train_loss", []), label="Train Loss")
    plt.plot(history.get("val_loss", []), label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
