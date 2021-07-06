from bokeh.plotting import figure, show
from wellcomeml.viz.palettes import Wellcome33
from bokeh.io import output_notebook
output_notebook()


def visualize_clusters(reduced_points: list):

    """
    Visualises clusters and shows basic information
    Args:
        reduced_points: list
        List of list of reduced points. Available at cluster.reduced_points

    Returns:
        p

    """

    blue_wellcome = str(Wellcome33[0])
    TOOLS = "hover, crosshair, pan, wheel_zoom, zoom_in, zoom_out, box_zoom, undo, redo, reset, tap, save, box_select, poly_select, lasso_select, "
    TOOLTIPS = [("index", "$index"), ("(x,y)", "($x, $y)"), ]

    p = figure(tools=TOOLS, tooltips=TOOLTIPS)

    p.scatter(reduced_points[:, 0], reduced_points[:, 1], radius=0.30,
              fill_color=blue_wellcome, line_color=None, alpha=0.5)

    show(p)