import matplotlib.pyplot as plt


def apply_plot_settings(axes, log=False, **kwargs):
    """Applies common settings."""
    axes.autoscale()
    if log:
        axes.set_xscale('log')
        axes.set_yscale('log')


def get_axes(axes, **kwargs):
    """If None, creates a new plot, otherwise returns its argument."""
    if axes is None:
        fig, axes = plt.subplots(figsize=(10, 8))
        fig.set_tight_layout(True)
        fig.show()
        apply_plot_settings(axes, **kwargs)
    return axes


def filter_kwargs_plot(kwargs):
    """Filters out keys unknown to plot function."""
    known_keys = {'color', 'linestyle', 'ls', 'linewidth', 'lw', 'marker', 'markersize', 'label'}
    return {key: value for key, value in kwargs.items() if key in known_keys}


def my_plot(x, y, axes=None, marker='.', **kwargs):
    axes = get_axes(axes)
    axes.plot(x, y, marker=marker, markersize=10, **filter_kwargs_plot(kwargs))
    handles = axes.get_legend_handles_labels()[0]
    if handles:
        axes.legend().set_draggable(True)
    return axes


def my_scatter(x, y, axes=None, marker='.', **kwargs):
    axes = get_axes(axes)
    axes.scatter(x, y, marker=marker, s=10, **filter_kwargs_plot(kwargs))
    handles = axes.get_legend_handles_labels()[0]
    if handles:
        axes.legend().set_draggable(True)
    return axes
