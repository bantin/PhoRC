import numpy as np
from sklearn.feature_extraction import img_to_graph
import plotly.express as px
import dash
from dash import dcc, html, callback_context
from scipy.ndimage import gaussian_filter
from dash.dependencies import Input, Output, State


import plotly.graph_objects as go
from plotly.subplots import make_subplots

import h5py
import sys

sys.path.append('../')
import grid_utils as util
import subtract_utils
import os


def _ask_user_for_path():
    user_input = input("Enter the path of your file: ")
    assert os.path.exists(user_input), "I did not find the file at, "+str(user_input)
    print("Hooray we found your file!")
    return user_input


def _load_data(path):
    assert os.path.exists(path), "I did not find the file at, " + str(path)

    pscs, stim_mat, powers, targets = util.load_h5_data(path)

    # power, x, y, z, time, repetitions
    raw_tensor = util.make_psc_tensor_multispot(pscs, powers, targets, stim_mat)

    # check if this is multispot data. If so, display map by running lasso
    if np.sum(stim_mat[:,0] > 0) > 1:
        raw_lasso_resp = util.circuitmap_lasso_cv(stim_mat, pscs)[0]
        raw_map = raw_lasso_resp.reshape(raw_tensor.shape[0:-2])
    else:
        raw_map = subtract_utils.traces_tensor_to_map(raw_tensor)

    num_powers = raw_map.shape[0]
    num_planes = raw_map.shape[-1]
    raw_map = np.moveaxis(raw_map, 0, -1)
    raw_map = raw_map.reshape(raw_map.shape[0], raw_map.shape[1],
        raw_map.shape[2] * raw_map.shape[3])

    return raw_tensor, raw_map, num_powers, num_planes

def save_data(traces, path):
    with h5py.File(path, 'a') as f:
        f.create_dataset('traces', data=traces)



## all functions with callbacks go here!
app = dash.Dash(__name__)
    

@app.callback(
    # Output('relay-dump','children'),
    Output('tracefig','figure'),
    Input('map_fig','clickData'),
    Input('tracefig', 'clickData')
)
def plot_traces(map_click, trace_click):
    global tracefig
    global map_fig

    timesteps=900
    colors = px.colors.qualitative.Dark24

    tracefig = make_subplots(rows=2, cols=1)
    if not map_click and not trace_click:
        raise dash.exceptions.PreventUpdate

    # determine which input caused the callback
    context = dash.callback_context
    print(context.triggered_id)

    # update displayed traces for single point on map
    if map_fig:

        y = map_click["points"][0]['x']
        x = map_click["points"][0]['y']

        print(x, y)
        clicked_plot_num = map_click["points"][0]['curveNumber']
        print(clicked_plot_num)

        plane_idx = clicked_plot_num // num_powers
        power_idx = clicked_plot_num % num_powers


        # plot clicked traces
        clicked_traces = raw_tensor[power_idx, x, y, plane_idx, :].reshape(-1, 900)
        for trace_idx, trace in enumerate(clicked_traces):
            tracefig.add_trace(
                go.Scatter(
                    x=np.arange(timesteps),
                    y=trace,
                    line=dict(color=colors[trace_idx])
                ),
                row=1,
                col=1
            )

    if context.triggered_id == 'tracefig':
        print(trace_click)

        # figure out which point in grid is selected
        y = map_click["points"][0]['x']
        x = map_click["points"][0]['y']
        clicked_plot_num = map_click["points"][0]['curveNumber']
        plane_idx = clicked_plot_num // num_powers
        power_idx = clicked_plot_num % num_powers
        trace_idx = trace_click['points'][0]['curveNumber']

        #Add trace to saved list
        saved_traces.append(raw_tensor[power_idx,
            x, y, plane_idx, trace_idx])

    # plot saved traces
    for trace_idx, trace in enumerate(saved_traces):
        tracefig.add_trace(
            go.Scatter(
                x=np.arange(timesteps),
                y=trace,
                line=dict(color=colors[trace_idx])
            ),
            row=2,
            col=1
        )

    # tracefig.update_yaxes(title_text="raw", range=[raw_min, raw_max], row=1, col=1)
    # tracefig.update_yaxes(title_text="est", range=[raw_min, raw_max], row=2, col=1)
    # tracefig.update_yaxes(title_text="subtracted", range=[subtracted_min, subtracted_max], row=3, col=1)
    # tracefig.update_yaxes(title_text="demixed", range=[subtracted_min, subtracted_max], row=4, col=1)

    tracefig.update_layout(
        height=800,
        # margin=dict(l=10, r=0, t=30, b=10),
    )
    return tracefig

@app.callback(
    Output('map_fig','figure'),
    Input('load-btn','n_clicks'),
    Input('load_path', 'value'),
    Input('save-btn', 'n_clicks'),
    Input('save_path', 'value'))
def load_new_map(load_nclicks, load_path, save_nclicks, save_path):
    global raw_map
    global num_powers
    global raw_tensor
    global num_planes
    global map_fig

    try:
        if dash.callback_context.triggered_id == 'load-btn' and load_path:
            print('loading new dataset: %s' % load_path)
            raw_tensor, raw_map, num_powers, num_planes = _load_data(load_path)

        elif dash.callback_context.triggered_id == 'save-btn' and save_path:
            save_data(saved_traces, save_path)
            print('Saved file to %s' % save_path)
    except dash.exceptions.MissingCallbackContextException:
        pass


    
    # update map figure
    map_fig=px.imshow(
        raw_map,
        facet_col=2,
        facet_col_wrap=num_powers,
        color_continuous_scale='jet',
        facet_col_spacing=0.03,
        facet_row_spacing=0.03,
        zmin=0,
        # zmax=20,
        origin='lower')

    map_fig.layout.coloraxis.showscale = True
    map_fig.update_layout(
        height=800,
    )
    return map_fig
    

# define all global vars and initialize
DEFAULT_PATH = "../data/masato/B6WT_AAV_hsyn_chrome2f_gcamp8/preprocessed/220308_B6_Chrome2fGC8_030822_Cell1_OpsPositive_A_planes_cmReformat.mat"
global saved_traces
global raw_tensor
global raw_map
global num_powers
global num_planes
saved_traces = []
raw_tensor, raw_map, num_powers, num_planes = _load_data(DEFAULT_PATH)
map_fig = load_new_map(None, None, None, None)


# placeholder until first callback
tracefig = make_subplots(rows=4, cols=1)
tracefig.update_layout(
    height=800,
)

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
                [dash.dcc.Graph(id="tracefig", figure=tracefig)], 
                style = {'display': 'inline-block', 'vertical_align': 'top'},
            ),
        ]),

        # second row
        html.Div([

            # inputs specifying dataset to load
            dcc.Input(id="load_path", type="text", placeholder="input_path",
                style={'marginRight':'10px'}),
            html.Button('Load dataset', id='load-btn', n_clicks=0,
                style={'marginRight': '20px'}),

            # inputs specifying dataset to save
            dcc.Input(id="save_path", type="text", placeholder="output_path",
                style={'marginRight':'10px'}),
            
            html.Button('Save', id='save-btn', n_clicks=0),
            html.Button('Undo', id='btn-nclicks-2', n_clicks=0), 
        ]),
        # html.Div(id='relay-dump'),
    ])



# Run app and display result inline in the notebook
if __name__ == '__main__':
    app.run_server(debug=True,port=8051)
