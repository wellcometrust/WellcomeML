from typing import Optional, Union

from wellcomeml.utils import throw_extra_import_message

from wellcomeml.viz.palettes import (Wellcome33, WellcomeBackground, WellcomeNoData)

required_modules = 'numpy,pandas,bokeh'
required_extras = 'core'

try:
    import numpy as np
    import pandas as pd

    from bokeh.io import output_notebook, reset_output
    from bokeh.models import Legend, Dropdown, ColumnDataSource, CustomJS
    from bokeh.plotting import figure, output_file, show
    from bokeh.layouts import column
    from bokeh.events import MenuItemClick
except ImportError as e:
    throw_extra_import_message(error=e, required_modules=required_modules, extras=required_extras)


def visualize_clusters(clustering, filter_list: Optional[list] = None,
                       texts: Optional[list] = None,
                       radius: float = 0.05,
                       alpha: float = 0.5,
                       plot_width: int = 1000, plot_height: int = 530,
                       output_in_notebook: bool = True,
                       output_file_path: Optional[str] = None,
                       palette: Union[list, str] = 'Wellcome33'):
    """
    This function creates a plot of the clusters

    Args:
        clustering: wellcomeml.ml.TextClustering instance
        filter_list: list
        texts: A list of texts to be displayed by the hover function
        radius: float, default: 0.05
        alpha: float, default: 0.5
        plot_width: int, default: 600
        plot_height: int, default: 600
        output_in_notebook: bool, default: True
        output_file_path: str, default: 'cluster_viz.html'
        palette: list, default: Wellcome33

    Returns:
        None (Prints a bokeh figure)
    """

    # Dataframe creation
    reduced_points = clustering.reduced_points
    data = pd.DataFrame(reduced_points)
    data = data.rename(columns={0: 'X', 1: 'Y'})
    data['cluster_id'] = clustering.cluster_ids
    data['cluster_id'] = data['cluster_id'].astype(str)
    data['Keywords'] = clustering.cluster_kws
    data['category'] = (filter_list if filter_list else ['All']*len(data))

    # Palette setting
    palette = (Wellcome33 if palette == 'Wellcome33' else palette)
    palette = [str(x) for x in palette]
    well_background = str(WellcomeBackground)
    clusters = list(data['cluster_id'])
    clusters = list(map(int, clusters))
    clusters_uniq = np.unique(clusters)
    data['colors'] = [(palette[x % len(palette)]
                       if x != -1 else str(WellcomeNoData)) for x in clusters]

    tools = ('hover, pan, wheel_zoom, zoom_in, zoom_out, reset, save, tap')
    tooltips = [("index", "$index"), ("(x,y)", "($x, $y)"),
                ("cluster", "@cluster_id"), ("keywords", "@Keywords")]

    if texts is not None:
        # Only gets the 60 characters of the text
        data['text'] = [text[:60] + '...' for text in texts]
        tooltips += [("text", "@text")]

    # DropDown Button
    dropdown_options = list(set([('All', 'All'), None] +
                                [(cat, cat) for i, cat in enumerate(sorted(
                                    data['category'].unique()), 2)]))
    dropdown = Dropdown(label='Category', button_type='default',
                        menu=dropdown_options, width=190, align="end")

    # Defines figure for plotting the clusters
    p = figure(title="Cluster visualization", toolbar_location="above",
               plot_width=plot_width, plot_height=plot_height,
               tools=tools, tooltips=tooltips,
               background_fill_color=well_background)

    R = []
    sources = []
    filtered_sources = []
    for x in clusters_uniq:
        data_cluster_id_unfiltered = data[data['cluster_id'] == str(x)]

        sources.append(ColumnDataSource(data_cluster_id_unfiltered))
        filtered_sources.append(ColumnDataSource(data_cluster_id_unfiltered))

        # Plots the cluster
        r = p.circle(x="X", y="Y", radius=radius, fill_alpha=alpha,
                     color="colors", source=filtered_sources[-1])

        R += [r]

    # JavaScript callback for the Dropdown Button
    callback = CustomJS(
        args=dict(sources=sources, filtered_sources=filtered_sources),
        code="""

    var data = []
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

    if (cat === 'All') {
        for (var i = 0; i < sources.length; i++) {
            data.push(sources[i].data);
        }
    } else {
        for (var i = 0; i < sources.length; i++) {
            let new_data =  generateNewDataObject(sources[i].data);
            for (var j = 0; j <= sources[i].data['category'].length; j++) {
                if (sources[i].data['category'][j] == cat) {
                    new_data = addRowToAccumulator(new_data, sources[i].data,
                                                   j);
                }
            }
            data[i] = new_data
        }
    }
     for (var i = 0; i < sources.length; i++) {
        filtered_sources[i].data = data[i]
        filtered_sources[i].change.emit()
    }
    """
    )
    dropdown.js_on_event(MenuItemClick, callback)

    # Plots the legend on two columns
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

    # Plots other extra annotations to the plot
    p.legend.title = "Cluster ID"
    p.legend.label_text_font_size = "11px"
    p.legend.background_fill_color = str(WellcomeBackground)
    p.legend.click_policy = "hide"
    p.min_border_left = 200

    # Output in notebook and new page
    reset_output()
    if output_in_notebook:
        output_notebook()

    if output_file_path:
        output_file(output_file_path)

    show(column(dropdown, p))
