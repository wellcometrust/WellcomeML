import random
import numpy as np
import pandas as pd
from bokeh.io import output_notebook, reset_output
from bokeh.models import Legend, Dropdown, ColumnDataSource, CustomJS
from bokeh.plotting import figure, output_file, show
from wellcomeml.viz.palettes import (Wellcome33,
                                     WellcomeBackground, WellcomeNoData)
from bokeh.layouts import column
from bokeh.events import MenuItemClick


def visualize_clusters(clustering, radius: float = 0.05, alpha: float = 0.5,
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
        alpha : float, default: 0.5
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
    # create 'category' column and generate it with random values
    data['category'] = pd.Series(random.choices(['Accepted', 'Rejected'],
                                                weights=[5, 1], k=len(data)))

    palette = [str(x) for x in palette]
    well_background = str(WellcomeBackground)
    clusters = list(data['cluster_id'])
    clusters = list(map(int, clusters))
    clusters_uniq = np.unique(clusters)
    data['colors'] = [(palette[x % len(palette)]
                       if x != -1 else str(WellcomeNoData)) for x in clusters]

    tools = ('hover, pan, wheel_zoom, zoom_in, zoom_out, reset, save')
    tooltips = [("index", "$index"), ("(x,y)", "($x, $y)"),
                ("cluster", "@cluster_id"), ("keywords", "@Keywords")]

    source = ColumnDataSource(data)
    dropdown_options = [('All', 'All'), None] + [
        (cat, cat) for i, cat in enumerate(sorted(data['category'].unique()),
                                           2)]
    # Generate dropdown widget
    dropdown = Dropdown(label='Category', button_type='default',
                        menu=dropdown_options)
    filtered_data = ColumnDataSource(data)

    # Callback
    callback = CustomJS(
        args=dict(unfiltered_data=source, filtered_data=filtered_data),
        code="""

    var data = unfiltered_data.data;
    var cat = cb_obj.item;

    function generateNewDataObject(oldDataObject){
        var newDataObject = {}
        for (var key of Object.keys(oldDataObject)){
            newDataObject[key] = [];
        }
        return newDataObject

    }

    function addRowToAccumulator(accumulator, dataObject, index) {
        for (var key of Object.keys(dataObject)){
            accumulator[key][index] = dataObject[key][index];
        }
        return accumulator;
    }

    if (cat === 'All'){
        data = unfiltered_data.data;
    } else {
        var new_data =  generateNewDataObject(data);
        for (var i = 0; i <= unfiltered_data.data['category'].length; i++){
            if (unfiltered_data.data['category'][i] == cat) {
                new_data = addRowToAccumulator(new_data, unfiltered_data.data,
                                               i);
            }
        }
        data = new_data;
    }

    filtered_data.data = data;
    filtered_data.change.emit();
    """
    )

    # Link actions
    dropdown.js_on_event(MenuItemClick, callback)

    p1 = figure(title="Cluster visualization", toolbar_location="above",
                plot_width=plot_width, plot_height=plot_height,
                tools=tools, tooltips=tooltips,
                background_fill_color=well_background,
                sizing_mode='scale_width')

    # p1.circle(x='X', y='Y', source=filtered_data, radius=radius,
    #          fill_alpha=alpha, color="colors")

    # p = figure(title="Cluster visualization", toolbar_location="above",
    #           plot_width=plot_width, plot_height=plot_height,
    #           tools=tools, tooltips=tooltips,
    #           background_fill_color=well_background,
    #           sizing_mode='scale_width')

    R = []
    for x in clusters_uniq:
        df = data[data['cluster_id'] == str(x)]
        r = p1.circle(x="X", y="Y", radius=radius, fill_alpha=alpha,
                      color="colors", source=df)
        R += [r]

    if len(clusters_uniq) > 36:
        median = len(R) // 2
        legend1 = Legend(items=[(str(s), [r]) for s, r in
                                zip(clusters_uniq[:median], R[:median])])
        legend2 = Legend(items=[(str(s), [r]) for s, r in
                                zip(clusters_uniq[median:], R[median:])])
        p1.add_layout(legend1, 'right')
        p1.add_layout(legend2, 'right')
    else:
        legend = Legend(items=[(str(s), [r]) for s, r in
                               zip(clusters_uniq, R)])
        p1.add_layout(legend, 'right')

    p1.legend.title = "Cluster ID"
    p1.legend.label_text_font_size = "10px"
    p1.legend.background_fill_color = str(WellcomeBackground)
    p1.legend.click_policy = "hide"

    reset_output()
    if output_in_notebook:
        output_notebook()
        show(column(dropdown, p1))
    else:
        output_file(output_file_path)
        show(column(dropdown, p1))
