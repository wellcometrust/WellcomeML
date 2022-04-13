from collections import defaultdict

import numpy as np
from bokeh.plotting import figure, show
from bokeh.io import output_notebook, output_file, reset_output
from bokeh.models import LabelSet, ColumnDataSource


from wellcomeml.viz.palettes import WellcomeBackground
from wellcomeml.viz.colors import NAMED_COLORS_DICT

A = 'test'

def plot_heatmap(co_occurrence,
                 concept_order=None,
                 file="heatmap.html",
                 notebook=True,
                 title="",
                 notebook_url='localhost:8888',
                 color='Blue Lagoon',
                 metadata_to_display=[],
                 rectangle_size=0.9,
                 display_percentages=True):
    """
    Plots a nice bokeh heatmap between two sets of "concepts" using the Wellcome pallette.


    Args:
        co_occurrence(list): list of dictionaries containing, at least:
        "concept_1", "concept_2", and the value of the edge, between 0 and 1.

        For example:

        co_ocurrence = [
            {"concept_1": apple, "concept_2": fruit, "value": 0.2},
            {"concept_1": apple, "concept_2": vegetable, "value": 0.1},
            {"concept_1": banana, "concept_2": fruit, "value": 0.1}
        ]

        concept_order(list): A list of order of concepts to plot, if none will chose arbitrary.
          Only applicable if list of "concepts_1" is the same as "concepts_2"
        file(str): A file to save the output
        color(str): Which color from the Wellcome pallete (see `wellcomeml.viz.colors`)
        metadata_to_display(list): List of 2-uples describing the legend and the key to display
          Exmaple, if the co_ocurrence matrix consist of the keys `concept_1`, `concept_2`,
          `value`, `legend`, one can send `metadata_to_display=[(This is a legend, legend)]`
        rectangle_size(int): Size of the rectangles to plot. Default: 0.9
        display_percentages(bool): Whether to show the percentages in the each rectangle as text

    Returns:
    """
    reset_output()

    x_names = []
    y_names = []
    alphas = []
    colors = []
    metadata = defaultdict(list)

    for row in co_occurrence:
        x_names.append(row['concept_1'])
        y_names.append(row['concept_2'])
        alphas.append(row['value'])

        for key, value in row.items():
            if key != ['concept_1', 'concept_2', 'concept_2']:
                metadata[key].append(value)

        colors.append(str(NAMED_COLORS_DICT[color]))

    data = {
        "x_name": x_names,
        "y_name": y_names,
        "alphas": alphas,
        "alpha_percentage": [100*alpha for alpha in alphas],
        "alpha_text": [f'{100*alpha:.0f}%' for alpha in alphas],
        "colors": colors
    }

    data = {**data, **metadata}

    data = ColumnDataSource(data=data)

    title = title
    tools = "hover,save,wheel_zoom"
    tooltips = [('Tags', '@y_name, @x_name'),
                ('Co-Ocurrence (percentage)', '@alpha_percentage')]

    if metadata_to_display:
        tooltips += [(legend, '@' + key) for legend, key in metadata_to_display]

    x_range = (concept_order if concept_order else list(reversed(np.unique(x_names))))
    # If the concepts are equal in rows and columns plot them in order, otherwise
    # just plot the unique y_concepts
    if set(x_names) == set(y_names):
        y_range = list(reversed(x_range))
    else:
        y_range = np.unique(y_names)

    p = figure(title=title, x_axis_location="below", tools=tools,
               x_range=x_range,
               y_range=y_range,
               tooltips=tooltips,
               background_fill_color=str(WellcomeBackground))

    if display_percentages:
        labels = LabelSet(x='x_name', y='y_name', text='alpha_text', text_font_size='16px',
                          text_align="center", source=data, render_mode='canvas')
        p.add_layout(labels)

    p.plot_width = 1000
    p.plot_height = 1000
    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "8pt"
    p.xaxis.major_label_orientation = np.pi / 3

    p.rect('x_name', 'y_name', rectangle_size, rectangle_size, source=data,
           color='colors', alpha='alphas', line_color=None,
           hover_line_color='black', hover_color='colors')

    if notebook:
        output_notebook()
    if file:
        output_file(file, title=title)

    show(p, notebook_url=notebook_url)
