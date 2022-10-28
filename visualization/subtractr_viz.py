import numpy as np
from sklearn.feature_extraction import img_to_graph
import plotly.express as px
import math, json
import dash
from dash import dcc, html, callback_context
from skimage import measure
from scipy.ndimage import gaussian_filter
from dash.dependencies import Input, Output, State
import scipy.sparse
import copy
from scipy.ndimage import convolve
import cv2
import plotly.graph_objects as go
import skimage.io
from plotly.subplots import make_subplots

import time
import matplotlib.pyplot as plt
import matplotlib.animation
import os


# load results dictionary
# use placeholder path for now
path = "../opsin_subtraction_experiments/220308_B6_Chrome2fGC8_030822_Cell2_opsPositive_A_planes_cunning-brattain-results.npz"
results = np.load(path)

# reshape map_img to create a single 4D array for all maps.
# axis 0 is power, so that we can index using power_slider value.
# axis 1 is both maps (raw, subtracted, etc) along with z_planes
# this is convenient for using the face_col argument of imshow
maps = np.stack([results['raw_map'],
    results['subtracted_map'], results['demixed_map']],
    axis=0,
)
num_maps, num_powers, num_xs, num_ys, num_zs = maps.shape

maps = np.transpose(maps, (1, 4, 0, 2, 3))
maps = np.reshape(maps, (num_powers, num_maps * num_zs, num_xs, num_ys))

unique_powers = np.unique(results['powers'])



# placeholder until first callback
tracefig = make_subplots(rows=4, cols=1)
tracefig.update_layout(
    height=800,
)

app = dash.Dash(__name__)


## all functions with callbacks go here!

@app.callback(
    # Output('relay-dump','children'),
    Output('graph4','figure'),
    # Output('button-value','children'),
    Input('map_fig','clickData'),
    Input('power_slider', 'value'))
def plot_traces(clickData, value):
    tracefig = make_subplots(rows=4, cols=1)
    if not clickData:
        raise dash.exceptions.PreventUpdate

    y = clickData["points"][0]['x']
    x = clickData["points"][0]['y']
    clicked_plot_num = clickData["points"][0]['curveNumber']
    plane_idx = clicked_plot_num // num_maps
    window_center=(x, y)

    colors = px.colors.qualitative.Dark24
    idx = 1

    raw_max = np.nanmax(results['raw_tensor'][value, x, y, plane_idx, :])
    raw_min = np.nanmin(results['raw_tensor'][value, x, y, plane_idx, :])
    subtracted_max = np.nanmax(results['subtracted_tensor'][value, x, y, plane_idx, :])
    subtracted_min = np.nanmin(results['subtracted_tensor'][value, x, y, plane_idx, :])
    for label in ['raw_tensor', 'est_tensor', 'subtracted_tensor', 'demixed_tensor']:
        traces = results[label][value, x, y, plane_idx, :]
        timesteps = traces.shape[-1]
        for trace_idx, trace in enumerate(traces):
            tracefig.add_trace(
                go.Scatter(
                    x=np.arange(timesteps),
                    y=trace,
                    line=dict(color=colors[trace_idx])
                ),
                row=idx,
                col=1
            )
        idx += 1
    tracefig.update_yaxes(title_text="raw", range=[raw_min, raw_max], row=1, col=1)
    tracefig.update_yaxes(title_text="est", range=[raw_min, raw_max], row=2, col=1)
    tracefig.update_yaxes(title_text="subtracted", range=[subtracted_min, subtracted_max], row=3, col=1)
    tracefig.update_yaxes(title_text="demixed", range=[subtracted_min, subtracted_max], row=4, col=1)

    tracefig.update_layout(
        height=800,
        # margin=dict(l=10, r=0, t=30, b=10),
    )
    return tracefig

@app.callback(
    Output('map_fig','figure'),
    Input('power_slider','value'))
def plot_maps(power_idx):
    map_fig=px.imshow(
        maps[power_idx,...],
        facet_col=0,
        facet_col_wrap=num_maps,
        color_continuous_scale='jet',
        facet_col_spacing=0.03,
        facet_row_spacing=0.03,
        zmin=0,
        zmax=20,
        origin='lower')

    map_fig.layout.coloraxis.showscale = True
    map_fig.update_layout(
        height=800,
    )
    return map_fig
map_fig = plot_maps(0) # initialize

app.layout = html.Div([

        # first row
        html.Div([
            # first column of first row
            html.Div(
                [dash.dcc.Graph(id="map_fig", figure=map_fig),],
                style={'display': 'inline-block', 'vertical_align': 'top'},
            ),
            # second column of first row
            html.Div(
                [dash.dcc.Graph(id="graph4", figure=tracefig)], 
                style = {'display': 'inline-block', 'vertical_align': 'top'},
            ),
        ]),

        # second row
        html.Div(dcc.Slider(0, num_powers-1, 1,
                id='power_slider',
                value=0,)
        ),
        # html.Div(id='relay-dump'),
    ])

# Run app and display result inline in the notebook
if __name__ == '__main__':
    app.run_server(debug=True,port=8051)
