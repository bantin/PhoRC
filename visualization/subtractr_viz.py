import os
import numpy as np
import plotly.express as px
import dash
import subtractr.utils as utils
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots



app = dash.Dash(__name__)

def _load_data(path):
    assert os.path.exists(path), f"Path {path} does not exist"
    results = np.load(path, allow_pickle=True)
    results = utils.sort_results(results)


    # reshape map_img to create a single 4D array for all maps.
    # axis 0 is power, so that we can index using power_slider value.
    # axis 1 is both maps (raw, subtracted, etc) along with z_planes
    # this is convenient for using the face_col argument of imshow.
    # If the results dictionary contains 'model_state'
    # then we also include CAVIaR weights.
    # if 'model_state' in results:
    #     weights = results['model_state']['mu']
    #     weights_map = weights.reshape(results['raw_map'].shape[1:])
    #     weights_map = np.broadcast_to(weights_map,
    #         (results['raw_map'].shape[0], *weights_map.shape))
    #     maps = np.stack(
    #         [results['raw_map'],
    #         results['subtracted_map'],
    #         results['demixed_map'], 
    #         weights_map],
    #         axis=0,
    #     )
    # else:
    #     maps = np.stack([results['raw_map'],
    #         results['subtracted_map'], results['demixed_map']],
    #         axis=0,
    #     )
    # num_maps, num_powers, num_xs, num_ys, num_zs = maps.shape

    # maps = np.transpose(maps, (1, 4, 0, 2, 3))
    # maps = np.reshape(maps, (num_powers, num_maps * num_zs, num_xs, num_ys))
    maps = [
        results['raw_map'],
        results['subtracted_map'],
        results['demixed_map'],
    ]
    if 'model_state' in results:
        weights = results['model_state']['mu']
        weights_map = weights.reshape(results['raw_map'].shape[1:])
        weights_map = np.broadcast_to(weights_map,
            (results['raw_map'].shape[0], *weights_map.shape))
        maps.append(weights_map)
    return maps, results


# setup before callbacks
global maps
global results
global tracefig
global num_powers
global num_maps

path = "../figures/full_pipeline/220308_B6_Chrome2fGC8_030822_Cell2/220308_B6_Chrome2fGC8_030822_Cell2_opsPositive_A_grid_cmReformat_with_nws_results.npz"
maps, results = _load_data(path)
num_powers = maps[0].shape[0]
num_maps = len(maps)

# placeholder until first callback
tracefig = make_subplots(rows=4, cols=1)
tracefig.update_layout(
    height=800,
)

## all functions with callbacks go here!
@app.callback(
    # Output('relay-dump','children'),
    Output('graph4','figure'),
    # Output('button-value','children'),
    Input('map_fig','clickData'),
    Input('power_slider', 'value'))
def plot_traces(clickData, value, max_start_idx=100, max_end_idx=700):
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

    
    raw_max = np.nanmax(results['raw_tensor'][value, x, y, plane_idx, :, max_start_idx:max_end_idx])
    raw_min = np.nanmin(results['raw_tensor'][value, x, y, plane_idx, :, max_start_idx:max_end_idx])
    subtracted_max = np.nanmax(results['subtracted_tensor'][value, x, y, plane_idx, :, max_start_idx:max_end_idx])
    subtracted_min = np.nanmin(results['subtracted_tensor'][value, x, y, plane_idx, :, max_start_idx:max_end_idx])
    for label in ['raw_tensor', 'est_tensor', 'subtracted_tensor', 'demixed_tensor']:
        traces = results[label][value, x, y, plane_idx, :]
        timesteps = traces.shape[-1]
        for trace_idx, trace in enumerate(traces):
            tracefig.add_trace(
                go.Scatter(
                    x=np.arange(timesteps),
                    y=trace,
                    line=dict(color=colors[trace_idx]),
                    showlegend=False,
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
    Input('load-btn','n_clicks'),
    Input('load_path', 'value'),
    Input('power_slider','value'),
)
def plot_maps(load_nclicks, load_path, power_idx):
    global maps
    global results
    global num_maps
    global num_powers

    try :
        if dash.callback_context.triggered_id == 'load-btn' and load_path:
            maps, results = _load_data(load_path)
            num_maps = len(maps)
            num_powers = maps[0].shape[0]
    except dash.exceptions.MissingCallbackContextException:
        pass

    # instead of using the facet_col argument,
    # we can create subplots and plot each map individually
    num_zs = maps[0].shape[-1]
    map_fig = make_subplots(rows=num_zs, cols=len(maps),
        shared_xaxes=True, shared_yaxes=True)

    for map_idx, map in enumerate(maps):
        maxval = np.nanmax(map[power_idx,...])
        for z_idx in range(num_zs):
            if z_idx == 0:
                cbar_dict = dict(
                    len=(1 / (num_maps + 1)),
                    lenmode='fraction',
                    y=1.05,
                    x=(1 / num_maps) * (map_idx + 0.5),
                    yanchor='middle',
                    orientation='h',
                )
                map_fig.add_trace(
                    go.Heatmap(
                        z=map[power_idx, :, :, z_idx],
                        zmin=0,
                        zmax=maxval,
                        colorbar=cbar_dict,
                        x=None,
                    ),
                    row=z_idx+1,
                    col=map_idx+1,
                )
            else:
                map_fig.add_trace(
                    go.Heatmap(
                        z=map[power_idx, :, :, z_idx],
                        zmin=0,
                        zmax=maxval,
                        colorbar=cbar_dict,
                        showscale=False,
                    ),
                    row=z_idx+1,
                    col=map_idx+1,
                )

    # map_fig=px.imshow(
    #     maps[power_idx,...],
    #     facet_col=0,
    #     facet_col_wrap=num_maps,
    #     color_continuous_scale='jet',
    #     facet_col_spacing=0.03,
    #     facet_row_spacing=0.03,
    #     zmin=0,
    #     # zmax=20,
    #     origin='lower')

    map_fig.layout.coloraxis.showscale = True
    map_fig.update_layout(
        height=800,
    )
    return map_fig
map_fig = plot_maps(None, None, 0)

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
        html.Div([
            html.Div(dcc.Slider(0, num_powers-1, 1,
                    id='power_slider',
                    value=0,)
            ),
        ]),

        # third row
        html.Div([
            # inputs specifying dataset to load
            dcc.Input(id="load_path", type="text", placeholder="input_path",
                style={'marginRight':'10px'}),
            html.Button('Load dataset', id='load-btn', n_clicks=0,
                style={'marginRight': '20px'}),
        ])
        
    ])


if __name__ == '__main__':
    app.run_server(debug=True,port=8051)
    

