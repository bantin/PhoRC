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

# def generate_plts(traces):

#     outfig3 = go.Figure()
#     timesteps = traces.shape[-1]
#     for trace in traces:
#         outfig3.add_trace(go.Scatter(
#             x=np.arange(0,timesteps),
#             y=trace,
#             name="Trace"),
#         )
#     print(traces.shape)
#     outfig3.update_layout(
#         autosize=False,
#          width=1400,
#         height=300,
#         margin=dict(l=50, r=0,  t=50, b=0),title_x=0.5)

#     outfig3.update_layout(title_text='Pixel trace')
#     outfig3.update_xaxes(title_text='Frame')
#     outfig3.update_yaxes(title_text="Intensity")
#     # outfig3.update_yaxes(title_text="Denoised", secondary_y=True)
#     return outfig3

# load results dictionary
# use placeholder path for now
path = "../mbcs_grids/figures/photocurrent_controls/220630_KynA/singlespot/220630_B6_Chrome2fGC8_IC_063022_Cell3_SingleSpot_Control_A_cmFormat_subtractr_caviar_results.npz"
results = np.load(path)

# reshape map_img so that powers and depths are all along the same axis,
# this is convenient for using the face_col argument of imshow
map_img = results['demixed_map']
num_powers, num_xs, num_ys, num_zs = map_img.shape
unique_powers = np.unique(results['powers'])



# tracefig=px.line([],title='Pixel trace')
tracefig = make_subplots(rows=1, cols=4)
tracefig.update_layout(
    width=1000,
    height=300,
    margin=dict(l=10, r=0, t=30, b=10),
)

app = dash.Dash(__name__)


## all functions with callbacks go here!

@app.callback(
    Output('relay-dump','children'),
    Output('graph4','figure'),
    # Output('button-value','children'),
    Input('map_fig','clickData'),
    Input('power_slider', 'value'))
def plot_traces(clickData, value):
    tracefig = make_subplots(rows=1, cols=4)
    if not clickData:
        raise dash.exceptions.PreventUpdate

    x = clickData["points"][0]['x']
    y = clickData["points"][0]['y']
    plane_idx = clickData["points"][0]['curveNumber']
    window_center=(x, y)

    idx = 1
    for label in ['raw_tensor', 'est_tensor', 'subtracted_tensor', 'demixed_tensor']:
        traces = results[label][value, x, y, plane_idx, :]
        timesteps = traces.shape[-1]
        for trace in traces:
            tracefig.add_trace(
                go.Scatter(
                    x=np.arange(timesteps),
                    y=trace,
                    name='trace'
                ),
                row=1,
                col=idx
            )
        idx += 1

    tracefig.update_layout(
        width=1000,
        height=300,
        margin=dict(l=10, r=0, t=30, b=10),
    )
    return json.dumps(window_center), tracefig

@app.callback(
    Output('map_fig','figure'),
    Input('power_slider','value'))
def plot_maps(power_idx):
    print(power_idx)
    map_fig=px.imshow(
        map_img[power_idx,...],
        facet_col=2,
        color_continuous_scale='jet',
        title='Main image',
        zmin=0,
        zmax=20,
        origin='lower')

    map_fig.layout.coloraxis.showscale = True
    map_fig.update_layout(
        margin=dict(l=20, r=20, t=30, b=0),title_x=0.5
    )
    return map_fig
map_fig = plot_maps(0) # initialize

app.layout = html.Div([
        html.Div([
            dash.dcc.Graph(id="map_fig", figure=map_fig)
        ],
            style={'width':'100%','display': 'inline-block','margin':'1vmax 0.5vmin 1vmax'},
            className='four columns'),
        html.Div(dcc.Slider(0, num_powers-1, 1,
                id='power_slider',
                value=0,
        )),
        html.Div(id='relay-dump'),
        html.Div(
        dash.dcc.Graph(id="graph4", figure=tracefig),style={'display': 'inline-block','margin-top':'1vmax',
        'margin-right':'20px','width':'100%'},className='four columns')
])



# Run app and display result inline in the notebook
if __name__ == '__main__':
    app.run_server(debug=True,port=8051)
