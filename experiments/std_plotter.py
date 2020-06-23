import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

from experiments.plotter import Plotter


class StdPlotter(Plotter):
    def __init__(self, vis, frequency=100, title=None, ylabel=None, xlabel=None, legend=None):
        super(StdPlotter, self).__init__(vis, frequency, title, ylabel, xlabel, legend)
        self.x = []
        self.y = []
        self.y_err = []
        self.ax = None
        self.fig = None

    def plot(self, x, y, y_err):
        super(StdPlotter, self).plot(x, y, y_err)
        self.x.append(x)
        self.y.append(y)
        self.y_err.append(y_err)

    def visualize_data(self):
        self.fig, self.ax = plt.subplots()
        self.ax.plot(self.x, self.y, '-')
        for i in range(len(self.y[0])):
            self.ax.fill_between(self.x,
                                 np.transpose(self.y)[i] - np.transpose(self.y_err)[i],
                                 np.transpose(self.y)[i] + np.transpose(self.y_err)[i],
                                 alpha=0.2)

    def save_figure_to_svg(self, dir, filename):
        self.visualize_data()
        self.fig.savefig(dir + filename, format="svg", bbox_inches='tight')

    def __smooth(self, ys, window=21, poly=5):
        yhat = savgol_filter(ys, window, poly)
        return yhat
