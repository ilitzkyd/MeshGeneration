from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt


class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices // 2

        self.im = ax.imshow(self.X[:, :, self.ind])
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        ax = self.ax
        self.im.set_data(self.X[:, :, self.ind])
        ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()


class IndexTrackerCompare(object):
    def __init__(self, ax1, X1, ax2, X2):
        self.ax1 = ax1
        self.ax2 = ax2
        self.X1 = X1
        self.X2 = X2
        rows, cols, self.slices = X1.shape
        self.ind = self.slices // 2

        self.im1 = ax1.imshow(self.X1[:, :, self.ind])
        self.im2 = ax2.imshow(self.X2[:, :, self.ind])
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        ax1 = self.ax1
        self.im1.set_data(self.X1[:, :, self.ind])
        self.im2.set_data(self.X2[:, :, self.ind])
        ax1.set_ylabel('slice %s' % self.ind)
        self.im1.axes.figure.canvas.draw()
        self.im2.axes.figure.canvas.draw()


def scroll_view(X, fig=None, ax=None):
    # Create a figure and axes if not provided
    if fig is None and ax is None:
        fig, ax = plt.subplots(1, 1)

    tracker = IndexTracker(ax, X)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()
    return ax


def scroll_view_compare(X1, X2, fig=None, ax1=None, ax2=None):
    # Create a figure and two axes if not provided
    if fig is None and ax1 is None and ax2 is None:
        fig = plt.figure()
        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2)

    tracker = IndexTrackerCompare(ax1, X1, ax2, X2)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()
    return ax1, ax2


def scatter_bw_img(bw_img, max_dots=5000):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    indices = np.argwhere(bw_img)
    n = indices.shape[0]
    delta = np.maximum(1, int(n / max_dots))

    # Generate hacky-axis-equal taken from https://stackoverflow.com/a/13701747/11756605
    max_range = np.array([indices[:, 0].max() - indices[:, 0].min(),
                          indices[:, 1].max() - indices[:, 1].min(),
                          indices[:, 2].max() - indices[:, 2].min()]).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (
            indices[:, 0].max() + indices[:, 0].min())
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (
            indices[:, 1].max() + indices[:, 1].min())
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (
            indices[:, 2].max() + indices[:, 2].min())
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')

    # Scatter plot non-zero indices of bw_img
    ax.scatter(indices[::delta, 0],
               indices[::delta, 1],
               indices[::delta, 2],
               s=2,
               c='r')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()
