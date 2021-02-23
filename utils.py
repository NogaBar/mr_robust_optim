import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_weight_graph(epochs, loss_lists, labels, name=''):
    epochs_array = np.arange(epochs)
    ax = plt.axes(xlabel='epoch', ylabel='weight', xticks=np.arange(0, epochs, 10),
                  yticks=np.arange(0, 10.0, 0.1))
    ax.set_title(name)
    y_min = float('inf')
    for loss_list, label in zip(loss_lists, labels):
        plt.plot(epochs_array, loss_list, label=label)
        min_loss = min(loss_list).cpu() if torch.is_tensor(min(loss_list)) else min(loss_list)
        y_min = min(y_min, min_loss)
    ax.legend()
    plt.grid(True, axis='y')
    plt.ylim(bottom=y_min-0.1, top=1.)
    plt.savefig('./images/%s.png'%name)
    plt.clf()


def plot_accuracy_graph(epochs, loss_lists, labels, name=''):
    epochs_array = np.arange(epochs)
    ax = plt.axes(xlabel='epoch', ylabel='accuracy', xticks=np.arange(0, epochs, 10),
                  yticks=np.arange(0, 10.0, 0.1))
    ax.set_title(name)
    y_min = float('inf')
    for loss_list, label in zip(loss_lists, labels):
        plt.plot(epochs_array, loss_list, label=label)
        y_min = min(y_min, min(loss_list))
    ax.legend()
    plt.grid(True, axis='y')
    plt.ylim(bottom=y_min-0.1, top=1.0)
    plt.savefig('./images/%s.png'%name)
    plt.clf()


def plot_loss_graph(epochs, loss_lists, labels, name=''):
    epochs_array = np.arange(epochs)
    ax = plt.axes(xlabel='epoch', ylabel='loss', xticks=np.arange(0, epochs, 10),
                  yticks=np.arange(0, 10.0, 0.1))
    ax.set_title(name)
    y_min = float('inf')
    for loss_list, label in zip(loss_lists, labels):
        plt.plot(epochs_array, loss_list, label=label)
        y_min = min(y_min, min(loss_list))
    ax.legend()
    plt.grid(True, axis='y')
    plt.ylim(bottom=y_min-0.1, top=4.0)
    plt.savefig('./images/%s.png'%name)
    plt.clf()