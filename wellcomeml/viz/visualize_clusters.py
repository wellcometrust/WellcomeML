from bokeh.plotting import figure, output_file, show
from wellcomeml.viz.palettes import Wellcome33, WellcomeBackground


def visualize_clusters(reduced_points: list):

    """
    Visualises clusters and shows basic information
    Args:
        reduced_points: list
        List of list of reduced points. Available at cluster.reduced_points

    Returns:
        None prints a bokeh figure to html file in new page

    """

    blue_wellcome = str(Wellcome33[0])
    well_background = str(WellcomeBackground)
    TOOLS = ('hover, pan, wheel_zoom, zoom_in, zoom_out, reset, save')
    TOOLTIPS = [("index", "$index"), ("(x,y)", "($x, $y)"), ]

    p = figure(title="Cluster visualisation", toolbar_location="above",
               tools=TOOLS, tooltips=TOOLTIPS,
               background_fill_color=well_background)

    p.scatter(reduced_points[0], reduced_points[1], radius=0.30,
              fill_color=blue_wellcome, line_color=None, alpha=0.5)

    output_file("cluster_viz.html")

    show(p)
