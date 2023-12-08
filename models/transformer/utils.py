import matplotlib.pylab as plt


def plot(arr: ndarray, x_str: str, y_str: str, title: str):
    plt.figure(figsize=(10, 6))
    plt.imshow(arr, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.xlabel(x_str)
    plt.ylabel(y_str)
    plt.title(title)
    plt.show()
    