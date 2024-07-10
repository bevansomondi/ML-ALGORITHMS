import matplotlib.pyplot as plt 


def plot_history(loss_history):
    ax = plt.subplot()
    ax.plot(loss_history)
    ax.set_xlabel("# epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss History")
    
    plt.grid()
    plt.show()

def plot_points(X, y, yhat):
    ax = plt.subplot()
    ax.scatter(X, y, label="data points")
    ax.scatter(X, yhat, label="predicted_probs")
    plt.legend()
    plt.show()
