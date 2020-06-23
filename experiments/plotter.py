import numpy as np
import visdom


class Plotter(object):
    def __init__(self, vis, frequency=100, title=None, ylabel=None, xlabel=None, legend=None):
        self.vis = vis
        self.win = None
        self.frequency = frequency
        self.title = title
        self.ylabel = ylabel
        self.xlabel = xlabel
        self.legend = legend
        self.values_array = []
        self.counter = 0
        self.reset()

    def reset(self):
        self.values_array.clear()
        self.counter = 0

    def log(self, episode, values):
        self.counter += 1
        self.values_array.append(values)
        if self.counter == self.frequency:
            self.plot(episode, np.mean(np.array(self.values_array), 0), np.std(np.array(self.values_array), 0))
            self.reset()

    def plot(self, episode, values, std):
        n_lines = len(values)
        if self.win is None:
            self.win = self.vis.line(X=np.arange(episode, episode + 1),
                                     Y=np.array([np.array(values)]),
                                     opts=dict(
                                         ylabel=self.ylabel,
                                         xlabel=self.xlabel,
                                         title=self.title,
                                         legend=self.legend))
        else:
            self.vis.line(X=np.array(
                [np.array(episode).repeat(n_lines)]),
                Y=np.array([np.array(values)]),
                win=self.win,
                update='append')
