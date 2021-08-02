import numpy as np
import pandas as pd
from bokeh.io import output_notebook, reset_output
from bokeh.models import ColumnDataSource, Legend
from bokeh.plotting import figure, output_file, show

from wellcomeml.viz.palettes import (Wellcome33,
                                     WellcomeBackground, WellcomeNoData)


def visualize_clusters(clustering, radius: float = 0.05, alpha: float = 0.8,
                       plot_width: int = 600, plot_height: int = 600,
                       output_in_notebook: bool = True,
                       output_file_path: str = 'cluster_viz.html',
                       palette: list = Wellcome33):

    """
    This function creates a plot of the clusters

    Parameters
    ----------
        clustering : class
        radius : float, default: 0.05
        alpha : float, default: 0.8
        plot_width : int, default: 600
        plot_height : int, default: 600
        output_in_notebook : bool, default: True
        output_file_path : str, default: 'cluster_viz.html'
        palette : list, default: Wellcome33

    Returns
    -------
        None
            Prints a bokeh figure
    """

    reduced_points = clustering.reduced_points
    data = pd.DataFrame(reduced_points)
    data = data.rename(columns={0: 'X', 1: 'Y'})
    data['cluster_id'] = clustering.cluster_ids
    data['cluster_id'] = data['cluster_id'].astype(str)
    data['Keywords'] = clustering.cluster_kws

    palette = [str(x) for x in palette]
    well_background = str(WellcomeBackground)
    clusters = list(data['cluster_id'])
    clusters = list(map(int, clusters))
    clusters_uniq = np.unique(clusters)
    data['colors'] = [(palette[x % len(palette)]
                       if x != -1 else str(WellcomeNoData)) for x in clusters]
    source = ColumnDataSource.from_df(data)

    tools = ('hover, pan, wheel_zoom, zoom_in, zoom_out, reset, save')
    tooltips = [("index", "$index"), ("(x,y)", "($x, $y)"),
                ("cluster", "@cluster_id"), ("keywords", "@Keywords")]

    p = figure(title="Cluster visualization", toolbar_location="above",
               plot_width=plot_width, plot_height=plot_height,
               tools=tools, tooltips=tooltips,
               background_fill_color=well_background,
               sizing_mode='scale_width')

    R = []
    for x in clusters_uniq:
        df = data[data['cluster_id'] == str(x)]
        r = p.circle(x="X", y="Y", color="colors", source=df)
        R += [r]

    if len(clusters_uniq) > 36:
        median = len(R) // 2
        legend1 = Legend(items=[(str(s), [r]) for s, r in
                                zip(clusters_uniq[:median], R[:median])])
        legend2 = Legend(items=[(str(s), [r]) for s, r in
                                zip(clusters_uniq[median:], R[median:])])
        p.add_layout(legend1, 'right')
        p.add_layout(legend2, 'right')
    else:
        legend = Legend(items=[(str(s), [r]) for s, r in
                               zip(clusters_uniq, R)])
        p.add_layout(legend, 'right')

    p.legend.title = "Cluster ID"
    p.legend.label_text_font_size = "10px"
    p.legend.background_fill_color = str(WellcomeBackground)
    p.legend.click_policy = "hide"

    reset_output()
    if output_in_notebook:
        output_notebook()
        show(p)
    else:
        output_file(output_file_path)
        show(p)
