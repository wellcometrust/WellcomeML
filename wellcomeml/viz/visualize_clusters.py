from bokeh.plotting import figure, output_file, show
from bokeh.io import output_notebook, reset_output
from bokeh.transform import factor_cmap
from bokeh.models import CDSView, ColumnDataSource, IndexFilter
from wellcomeml.viz.palettes import Wellcome33, WellcomeBackground


def visualize_clusters(reduced_points: list, radius: float,
                       alpha: float, output_in_notebook: bool,
                       output_file_path: str = 'cluster_viz.html'):

    """
    Visualises clusters and shows basic information
    Args:
        reduced_points: list
        List of list of reduced points. Available at cluster.reduced_points
        radius: float
        Radius of the circles in the scatter plot
        alpha: float
        Percentage of fillness when coloring the circle
        output_in_notebook: bool
        Boolean variable, it True the plot is in the notebook,
        if False the plot is in a new html page
        output_file_path: str = 'cluster_viz.html'
        By default equal to 'cluster_viz.html'

    Returns:
        None prints a bokeh figure to html file in new page

    """

    source = ColumnDataSource(reduced_points)
    clusters = list(reduced_points.ClusterID.unique())
    Wellcome33_palette = [str(x) for x in Wellcome33]
    well_background = str(WellcomeBackground)
    tools = ('hover, pan, wheel_zoom, zoom_in, zoom_out, reset, save')
    tooltips = [("index", "$index"), ("(x,y)", "($x, $y)"),
                ("cluster", "@ClusterID"), ("keywords", "@Keywords"),
                ("text", "@Text")]

    p = figure(title="Cluster visualisation", toolbar_location="above",
               tools=tools, tooltips=tooltips,
               background_fill_color=well_background)

    p.scatter(x='X', y='Y', radius=radius, source=source,
              color=factor_cmap('ClusterID', Wellcome33_palette, clusters),
              line_color=None, alpha=alpha)

    reset_output()
    if output_in_notebook == True:
        output_notebook()
        show(p)
    else:
        output_file(output_file_path)
        show(p)
