import matplotlib.pyplot as plt

def apply_plot_settings(axes):
    """Applies common settings."""
    axes.autoscale()
    axes.set_xscale('log')
    axes.set_yscale('log')
    axes.set_xlabel('N')
    axes.set_ylabel('Error')


def get_axes(axes):
    """If None, creates a new plot, otherwise returns its argument."""
    if axes is None:
        _, axes = plt.subplots()
        apply_plot_settings(axes)
    return axes


def filter_kwargs_plot(kwargs):
    """Filters out keys unknown to plot function."""
    known_keys = {'color', 'linestyle', 'linewidth',  'marker', 'markersize', 'label'}
    return {key: value for key, value in kwargs.items() if key in known_keys}


