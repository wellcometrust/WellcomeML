import pandas as pd
from bokeh.plotting import figure, output_file, show
from bokeh.io import output_notebook, reset_output
from bokeh.models import ColumnDataSource
from wellcomeml.viz.palettes import (Wellcome33,
                                     WellcomeBackground, WellcomeNoData)


def visualize_clusters(clustering, radius: float,
                       alpha: float, output_in_notebook: bool,
                       output_file_path: str = 'cluster_viz.html'):

    """
    Visualises clusters and shows basic information
    Args:
        clustering: class
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

    reduced_points = clustering.reduced_points
    data = pd.DataFrame(reduced_points)
    data = data.rename(columns={0: 'X', 1: 'Y'})
    data['cluster_id'] = clustering.cluster_ids
    data['cluster_id'] = data['cluster_id'].astype(str)
    data['Keywords'] = clustering.cluster_kws

    Wellcome33_palette = [str(x) for x in Wellcome33]
    well_background = str(WellcomeBackground)
    clusters = list(data['cluster_id'])
    clusters = list(map(int, clusters))
    data['colors'] = [(Wellcome33_palette[x % 33]
                       if x != -1 else str(WellcomeNoData)) for x in clusters]
    source = ColumnDataSource.from_df(data)

    tools = ('hover, pan, wheel_zoom, zoom_in, zoom_out, reset, save')
    tooltips = [("index", "$index"), ("(x,y)", "($x, $y)"),
                ("cluster", "@cluster_id"), ("keywords", "@Keywords")]

    p = figure(title="Cluster visualisation", toolbar_location="above",
               tools=tools, tooltips=tooltips,
               background_fill_color=well_background)

    p.scatter(x='X', y='Y', radius=radius, source=source,
              color='colors', line_color=None, alpha=alpha)

    reset_output()
    if output_in_notebook:
        output_notebook()
        show(p)
    else:
        output_file(output_file_path)
        show(p)
