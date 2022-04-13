########################################################################################################################
# Visualization application with DASH+Plotly
########################################################################################################################

import json
import cv2
import pandas as pd
import numpy as np
import dash
import string
from _plotly_utils.colors import n_colors
from dash import dcc, Output, Input, State
from dash import html
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from dash_extensions import Keyboard
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix

from application.default_figures import *
from data_utils.DataContainer import DataContainer
from data_utils.TestSet import TestSet

app = dash.Dash(__name__, title="DABrainVisApp")

MAX_VALUE_ASSD = 362
MODELS_SIMPLE1 = ["XNet_T2_relu", "XNet_T2_leaky", "XNet_T2_selu"]
MODELS_SIMPLE2 = ["XNet_T1_relu", "XNet_T1_leaky", "XNet_T1_selu"]
MODELS_SIMPLE_CG = ["CG_XNet_T1_relu", "CG_XNet_T2_relu"]
MODELS_BASELINE = [*MODELS_SIMPLE1, *MODELS_SIMPLE2, *MODELS_SIMPLE_CG]
MODELS_SEGMS2T = ["SegmS2T_GAN1_relu", "SegmS2T_GAN2_relu", "SegmS2T_GAN5_relu",
                  "CG_SegmS2T_GAN1_relu", "CG_SegmS2T_GAN2_relu", "CG_SegmS2T_GAN5_relu"]
MODELS_GAN_XNET = ["GAN_1+XNet_T1_relu", "GAN_2+XNet_T1_relu", "GAN_5+XNet_T1_relu",
                   "GAN_1+CG_XNet_T1_relu", "GAN_2+CG_XNet_T1_relu", "GAN_5+CG_XNet_T1_relu"]
MODELS_DA = [*MODELS_SEGMS2T, *MODELS_GAN_XNET]
MODELS = [*MODELS_BASELINE, *MODELS_DA]
MODELS_MAPPING = dict(zip(MODELS, list(string.ascii_uppercase)[0:len(MODELS)]))  # TODO: use mapping!
MODELS_CG = [*MODELS_SIMPLE_CG,
             "CG_SegmS2T_GAN1_relu", "CG_SegmS2T_GAN2_relu", "CG_SegmS2T_GAN5_relu",
             "GAN_1+CG_XNet_T1_relu", "GAN_2+CG_XNet_T1_relu", "GAN_5+CG_XNet_T1_relu"]
MODELS_NOT_CG = [*MODELS_SIMPLE1, *MODELS_SIMPLE2,
                 "SegmS2T_GAN1_relu", "SegmS2T_GAN2_relu", "SegmS2T_GAN5_relu",
                 "GAN_1+XNet_T1_relu", "GAN_2+XNet_T1_relu", "GAN_5+XNet_T1_relu"]
MODELS_BEST = ["XNet_T2_relu", "CG_XNet_T2_relu",
               "SegmS2T_GAN5_relu", "SegmS2T_GAN1_relu", "GAN_2+XNet_T1_relu", "GAN_1+XNet_T1_relu",
               "CG_SegmS2T_GAN2_relu", "GAN_2+CG_XNet_T1_relu"]

METRICS = ["DSC", "ASSD", "ACC", "TPR", "TNR"]

DATA_PATH = "/tf/workdir/data/VS_segm/test_processed"

# header
load_data = html.Div(
    id="load_data",
    children=[html.Div(children=[dbc.Button('Load Data', color='primary', size='sm', id='button-load-data')],
                       style={'width': '30%', 'display': 'inline-block', 'textAlign': 'center',
                              'verticalAlign': 'middle'}),
              html.Div(children=[dcc.Loading(id='loading-data', type="default",
                                             children=html.Div(id='loading-data-output',
                                                               style={'color': 'lightgray'}))],
                       style={'width': '70%', 'display': 'inline-block', 'textAlign': 'left',
                              'verticalAlign': 'middle'}),
              dcc.Store(id='df-total'),
              dcc.Store(id='df-signature'),
              dcc.Store(id='df-volume'),
              ],
    style={'width': '15%',
           'verticalAlign': 'middle',
           'display': 'inline-block'})
header1 = html.Div(
    id='header-1',
    children=html.H1(children='Segmentation Prediction Evaluation',
                     style={'textAlign': 'center', 'width': '66%', 'color': 'white', 'verticalAlign': 'middle'}),
    style={'width': '70%', 'display': 'inline-block'})
github_link = html.Div(
    id='github_link',
    children=[dbc.Button("Github Docu",
                         id='github_link_redirect',
                         color='light',
                         outline=True,
                         href='https://github.com/CarolineMagg/VA_brain_tumor#readme',
                         style={'verticalAlign': 'middle', 'textAlign': 'left',
                                'display': 'inline-block', 'width': '45%'}),
              html.Img(id="logo", src=app.get_asset_url("dash-logo-border.png"),
                       width='55%',
                       style={'verticalAlign': 'middle', 'textAlign': 'right',
                              'display': 'inline-block'})
              ],
    style={'width': '15%', 'display': 'inline-block', 'verticalAlign': 'middle'})
div_header = html.Div(
    id='header',
    children=[load_data,
              header1,
              github_link],
    style={'width': '100%',
           'backgroundColor': 'black'})

# control panel
control_dataset_mask = html.Div(
    children=[
        html.Div(children=html.Label("Dataset:"),
                 style={'width': '10%',
                        'textAlign': 'right',
                        'paddingTop': '3%',
                        'paddingRight': '1%',
                        'display': 'table-cell'}),
        html.Div(children=[dcc.RadioItems(id="radioitem-dataset",
                                          options=[{'label': 'only tumor',
                                                    'value': 'only_tumor'},
                                                   {'label': 'all',
                                                    'value': 'all'}],
                                          value='only_tumor')],
                 style={'width': '15%',
                        'textAlign': 'left',
                        'display': 'table-cell'}),
        html.Div([Keyboard(id="shortcut-mask")])
    ],
    style={'width': '25%',
           'textAlign': 'left',
           'display': 'table-cell'}
)

control_error_metric_dataset = html.Div(
    children=[html.Div(children=html.Label("Error Metric:"),
                       style={'width': '10%',
                              'textAlign': 'left',
                              'paddingLeft': '1%',
                              'verticalAlign': 'middle',
                              'display': 'table-cell'}),
              html.Div(children=dcc.Dropdown(id='dropdown-error-metric',
                                             options=[{'label': 'DSC (DiceSimilarityCoefficient)',
                                                       'value': 'DSC'},
                                                      {'label': 'ASSD (AverageSymmetricSurfaceDistance)',
                                                       'value': 'ASSD'},
                                                      {'label': 'ACC (Accuracy)',
                                                       'value': 'ACC'},
                                                      {'label': 'TPR (TruePositiveRate)',
                                                       'value': 'TPR'},
                                                      {'label': 'TNR (TrueNegativeRate)',
                                                       'value': 'TNR'}],
                                             value='DSC'
                                             ),
                       style={'width': '35%',
                              'verticalAlign': 'middle',
                              'display': 'table-cell'}),
              control_dataset_mask],
    style={'display': 'table',
           'width': '97%'})
control_model = html.Div(
    children=[html.Div(children=html.Label("Model:"),
                       style={'width': '10%',
                              'textAlign': 'left',
                              'verticalAlign': 'middle',
                              'paddingLeft': '1%',
                              'display': 'table-cell'}),
              html.Div(children=dcc.Dropdown(id="dropdown-model",
                                             options=[{"label": "All", "value": "All"}] +
                                                     [{"label": "Baseline", "value": "Baseline"}] +
                                                     [{"label": "DA", "value": "DA"}] +
                                                     [{"label": "Best", "value": "Best"}] +
                                                     [{"label": "CG", "value": "CG"}] +
                                                     [{"label": "NOT_CG", "value": "NOT_CG"}] +
                                                     [{"label": "SegmS2T", "value": "SegmS2T"}] +
                                                     [{"label": "Gen+Segm", "value": "Gen+Segm"}] +
                                                     [{"label": k, "value": k} for k in MODELS],
                                             placeholder="Select a list of models...",
                                             value=["DA"],
                                             multi=True),
                       style={'width': '85%',
                              'display': 'table-cell',
                              'font-size': '85%'}),
              html.Div(children=[html.Button(id='submit-model', n_clicks=0, children="Apply")],
                       style={'width': '5%',
                              'display': 'table-cell',
                              'verticalAlign': 'middle',
                              'textAlign': 'center',
                              'align': 'center'})],
    style={'display': 'table',
           'width': '97%'})
div_control_panel_heatmap = html.Div(
    children=[control_error_metric_dataset,
              html.Br(),
              control_model],
    style={'display': 'table-cell',
           'width': '66%',
           'border-width': '0px 1px 0px 2px', 'border-color': 'black', 'border-style': 'solid'})

control_prediction = html.Div(
    children=[html.Div(children=html.Label("Mask:"),
                       style={'width': '10%',
                              'textAlign': 'left',
                              'paddingLeft': '1%',
                              'display': 'table-cell'}),
              html.Div(children=dcc.RadioItems(id='radioitem-slice-view',
                                               options=[{'label': 'sum',
                                                         'value': 'sum'},
                                                        {'label': 'subtraction',
                                                         'value': 'subtraction'}],
                                               value='sum'),
                       style={'width': '30%',
                              'textAlign': 'left',
                              'display': 'table-cell'}),
              html.Div(children=html.Label("Opacity:"),
                       style={'width': '15%',
                              'textAlign': 'left',
                              'paddingLeft': '1%',
                              'display': 'table-cell'}),
              html.Div(children=dcc.Slider(id='slider-mask-opacity',
                                           min=0,
                                           max=1,
                                           step=0.1,
                                           value=0.8,
                                           tooltip={"placement": "bottom", "always_visible": True}
                                           ),
                       style={'width': '45%',
                              'verticalAlign': 'middle',
                              'display': 'table-cell'})],
    style={'display': 'table',
           'width': '97%'})
control_gt = html.Div(
    children=[html.Div(children=html.Label("GT:"),
                       style={'width': '10%',
                              'textAlign': 'left',
                              'paddingLeft': '1%',
                              'display': 'table-cell'}),
              html.Div(children=[dcc.RadioItems(id='radioitem-gt-toggle',
                                                options=[{'label': 'show',
                                                          'value': 'show'},
                                                         {'label': 'hide',
                                                          'value': 'hide'}],
                                                value='hide')],
                       style={'width': '90%',
                              'textAlign': 'left',
                              'display': 'table-cell'})],
    style={'display': 'table',
           'width': '97%'})
control_gt_2 = html.Div(
    id="control_gt_2",
    children=[html.Div(children=[],
                       style={'width': '10%',
                              'paddingLeft': '1%',
                              'textAlign': 'left',
                              'display': 'table-cell'}),
              html.Div(children=dcc.RadioItems(id='radioitem-gt-type',
                                               options=[{'label': 'contour',
                                                         'value': 'contour'},
                                                        {'label': 'mask',
                                                         'value': 'mask'}],
                                               value="contour"),
                       style={'width': '30%',
                              'textAlign': 'left',
                              'display': 'table-cell'}),
              html.Div(children=html.Label("Opacity:"),
                       style={'width': '15%',
                              'textAlign': 'left',
                              'paddingLeft': '1%',
                              'display': 'table-cell'}),
              html.Div(children=dcc.Slider(id='slider-gt-opacity',
                                           min=0,
                                           max=1,
                                           step=0.1,
                                           value=0.9,
                                           tooltip={"placement": "bottom", "always_visible": True}
                                           ),
                       style={'width': '45%',
                              'verticalAlign': 'middle',
                              'display': 'table-cell'})],
    style={'display': 'none',
           'width': '97%'})

div_slice_control_panel = html.Div(
    children=[control_prediction,
              html.Br(),
              control_gt,
              html.Br(),
              control_gt_2],
    style={'display': 'table-cell',
           'width': '33%',
           'border-width': '0px 2px 0px 1px', 'border-color': 'black', 'border-style': 'solid'})
control_panel = html.Div(
    id="control_panel",
    style={'width': '100%',
           'backgroundColor': 'grey',
           'display': 'table'},
    children=[div_control_panel_heatmap,
              div_slice_control_panel,
              dcc.Store(id='df-metric-overview'),
              dcc.Store(id='df-metric-detail'),
              dcc.Store(id='df-feature-overview'),
              dcc.Store(id='df-feature-detail'),
              dcc.Store(id='dict-clusters-metrics'),
              dcc.Store(id='df-heatmap-overview-overlay'),
              dcc.Store(id='df-heatmap-detail-overlay'),
              dcc.Store(id='dict-bin-mapping'),
              dcc.Store(id='dict-bin-mapping2'),
              dcc.Store(id='dict-bin-mapping3'),
              dcc.Store(id='dict-patient-id'),
              dcc.Store(id='dict-slice-id'),
              dcc.Store(id='clickdata-parcats-overview'),
              dcc.Store(id='clickdata-parcats-detail'),
              dcc.Store(id='json-selected-models'),
              dcc.Store(id='dict-slice-data')])

button1 = html.Div(children=html.Button(id='reset-button-overview', n_clicks=0, children="Reset"),
                   style={'width': '30%',
                          'verticalAlign': 'middle',
                          'display': 'inline-block',
                          'textAlign': 'left'})
h2_overview = html.Div(children=html.H2("All Patients", id='header-overview', style={'textAlign': 'center'}),
                       style={'width': '40%',
                              'display': 'inline-block',
                              'verticalAlign': 'middle',
                              'textAlign': 'center'})

button2 = html.Div(children=html.Button(id='reset-button-detail', n_clicks=0, children="Reset"),
                   style={'width': '30%',
                          'verticalAlign': 'middle',
                          'display': 'inline-block',
                          'textAlign': 'left'})
h2_detail = html.Div(children=html.H2("Patient", id='header-detail', style={'textAlign': 'center'}),
                     style={'width': '40%',
                            'display': 'inline-block',
                            'verticalAlign': 'middle',
                            'textAlign': 'center'})

empty = html.Div(children=[],
                 style={'width': '30%',
                        'verticalAlign': 'middle',
                        'display': 'inline-block',
                        'textAlign': 'right'})

sub_headers = html.Div(
    id='sub-headers',
    children=[html.Div(children=[empty, h2_overview, button1],
                       style={'width': '33.3%', 'display': 'table-cell',
                              'border-width': '2px 1px 2px 2px', 'border-color': 'black', 'border-style': 'solid'}),
              html.Div(children=[empty, h2_detail, button2],
                       style={'width': '33.3%', 'display': 'table-cell',
                              'border-width': '2px 1px 2px 1px', 'border-color': 'black', 'border-style': 'solid'}),
              html.Div(children=html.H2("Slice", id='header-slice', style={'textAlign': 'center'}),
                       style={'width': '33.3%', 'display': 'table-cell',
                              'border-width': '2px 2px 2px 1px', 'border-color': 'black', 'border-style': 'solid'})],
    style={'width': '100%',
           'backgroundColor': 'darkgray',
           'borderColor': 'black',
           'display': 'table'})

# heatmaps
heatmap_1 = html.Div(
    children=[dcc.Graph(id='heatmap-overview',
                        figure=fig_no_data_available),
              html.Div(children=dcc.RangeSlider(id='heatmap-overview-slider',
                                                min=0,
                                                max=1,
                                                step=0.01,
                                                value=[0, 1],
                                                marks=None,
                                                tooltip={'placement': 'bottom',
                                                         'always_visible': True},
                                                allowCross=False),
                       style={'textAlign': 'center'})],
    style={'display': 'table-cell',
           'width': '33.3%'})
heatmap_2 = html.Div(
    children=[dcc.Graph(id='heatmap-detail',
                        figure=fig_no_data_selected),
              dcc.RangeSlider(id='heatmap-detail-slider',
                              min=0,
                              max=1,
                              step=0.01,
                              value=[0, 1],
                              marks=None,
                              tooltip={'placement': 'bottom',
                                       'always_visible': True},
                              allowCross=False)],
    style={'display': 'table-cell',
           'width': '33.3%'})
slice_plot = html.Div(
    children=[html.Div(children=[dcc.Slider(id='slice-slider',
                                            className='rc-slider2',
                                            min=0,
                                            max=80,
                                            step=1,
                                            value=0,
                                            marks=None,
                                            tooltip={'always_visible': True},
                                            vertical=True)],
                       style={'width': '8%', 'height': '100%',
                              'display': 'inline-block', 'position': 'relative'}),
              html.Div(children=[dcc.Graph(id='slice-plot', figure=fig_no_slice_selected)],
                       style={'width': '92%', 'height': '100%', 'padding-bottom': '5%',
                              'display': 'inline-block', 'position': 'relative'})],
    style={'display': 'table-cell',
           'width': '33.3%'})
first_row = html.Div(
    id='heatmaps',
    style={'width': '100%',
           'display': 'table'},
    children=[heatmap_1,
              heatmap_2,
              slice_plot])

# parallel set plots
parcats_overview = html.Div(
    children=[dcc.Graph(id='parcats-overview',
                        clear_on_unhover=True,
                        figure=fig_no_data_available)],
    style={'display': 'table-cell',
           'width': '50%'})
parcats_detail = html.Div(
    children=[dcc.Graph(id="parcats-detail",
                        clear_on_unhover=True,
                        figure=fig_no_data_selected)],
    style={'display': 'table-cell',
           'width': '50%'})
parcats_control_general = html.Div(
    children=[
        html.Div(children=html.Label("Features:"),
                 style={'width': '10%',
                        'textAlign': 'left',
                        'paddingRight': '1%',
                        'paddingLeft': '1%',
                        'verticalAlign': 'middle',
                        'display': 'table-cell'}),
        html.Div(children=dcc.Dropdown(id='dropdown-feature',
                                       options=[{'label': 'Shape',
                                                 'value': 'shape'},
                                                {'label': 'Firstorder',
                                                 'value': 'firstorder'},
                                                {'label': 'Performance',
                                                 'value': 'performance'},
                                                {'label': 'Custom',
                                                 'value': 'custom'}, ],
                                       value="custom"
                                       ),
                 style={'width': '35%',
                        'textAlign': 'left',
                        'verticalAlign': 'middle',
                        'display': 'table-cell'}),
        html.Div(children=[html.Button(id='submit_theme', n_clicks=0, children="Theme",
                                       style={'color': 'white', 'backgroundColor': 'darkgray'}),
                           dcc.Store('theme', data=json.dumps("gray"))],
                 style={'width': '5%',
                        'textAlign': 'center',
                        'verticalAlign': 'middle',
                        'display': 'table-cell'})],
    style={'width': '50%',
           'verticalAlign': 'middle',
           'display': 'table-cell'})
parcats_control_pcd = html.Div(
    children=[
        html.Div(children=html.Label("PCD-Type (Slice):"),
                 style={'width': '15%',
                        'textAlign': 'left',
                        'paddingLeft': '1%',
                        'verticalAlign': 'middle',
                        'display': 'table-cell'}),
        html.Div(children=dcc.Dropdown(id='dropdown-pcd-cluster',
                                       options=[{'label': 'Slice',
                                                 'value': 'slice_pcd'},
                                                {'label': 'Model',
                                                 'value': 'model_pcd'}],
                                       value="slice_pcd"
                                       ),
                 style={'width': '20%',
                        'verticalAlign': 'middle',
                        'display': 'table-cell'}),
        html.Div(children=html.Label("Number of Clusters:", id="dropdown-pcd-label"),
                 style={'width': '15%',
                        'textAlign': 'left',
                        'paddingLeft': '1%',
                        'verticalAlign': 'middle',
                        'display': 'table-cell'}),
        html.Div(children=dcc.Dropdown(id='dropdown-pcd-type',
                                       options=[{'label': '2',
                                                 'value': '2'},
                                                {'label': '3',
                                                 'value': '3'},
                                                {'label': '4',
                                                 'value': '4'},
                                                {'label': '5',
                                                 'value': '5'}
                                                ],
                                       value=3
                                       ),
                 style={'width': '5%',
                        'verticalAlign': 'middle',
                        'display': 'table-cell'}),
        html.Div(style={'width': '1%',
                        'display': 'table-cell'})
    ],
    style={'width': '50%',
           'paddingLeft': '1%',
           'display': 'table-cell'})
parcats_control = html.Div(
    children=[parcats_control_general,
              parcats_control_pcd],
    style={'width': '100%',
           'backgroundColor': 'grey',
           'border-width': '2px 1px 2px 2px', 'border-color': 'black', 'border-style': 'solid',
           'display': 'table',
           'verticalAlign': 'middle',
           'textAlign': 'center'}
)

second_row = html.Div(
    id='parallel-set-plots',
    style={'width': '100%',
           'display': 'table',
           'backgroundColor': 'grey'},
    children=[parcats_overview,
              parcats_detail])

third_row = html.Div(
    id='third-row',
    style={'width': '100%',
           'height': '100px',
           'display': 'in-block'},
    children=html.Div()
)

# layout
app.layout = html.Div(
    children=[div_header,
              control_panel,
              sub_headers,
              first_row,
              parcats_control,
              second_row,
              third_row],
    style={'height': '100%',
           'width': '100%',
           'display': 'inline-block'})


def get_colorscale_tickvals(metric, slider_values, slider_max):
    # define colorscale and tickvals
    lookup_color = list(reversed([*px.colors.sequential.Plasma])) if metric == "ASSD" else [
        *px.colors.sequential.Plasma]
    steps = (slider_values[1] - slider_values[0]) / 9
    colorscale = []
    if slider_values[0] != 0:
        colorscale.append([0, lookup_color[0]])
    for idx, x in enumerate(np.arange(slider_values[0], slider_values[1], steps)):
        colorscale.append([x / slider_max, lookup_color[idx]])
    colorscale.append([slider_values[1] / slider_max, lookup_color[-1]])
    if slider_values[1] != 1:
        colorscale.append([1, lookup_color[-1]])
    tickvals = np.arange(0, slider_max, 20) if metric == "ASSD" else np.arange(0, np.nextafter(1.0, np.inf), 0.1)
    return colorscale, tickvals


def get_selected_model_list(models):
    models_selected = []
    if "All" not in models:
        models_selected = models
        if "Baseline" in models:
            models_selected += [m for m in MODELS_BASELINE]
            models_selected.remove("Baseline")
        if "DA" in models:
            models_selected += [m for m in MODELS_DA]
            models_selected.remove("DA")
        if "CG" in models:
            models_selected += [m for m in MODELS_CG]
            models_selected.remove("CG")
        if "NOT_CG" in models:
            models_selected += [m for m in MODELS_NOT_CG]
            models_selected.remove("NOT_CG")
        if "SegmS2T" in models:
            models_selected += [m for m in MODELS_SEGMS2T]
            models_selected.remove("SegmS2T")
        if "Best" in models:
            models_selected += [m for m in MODELS_BEST]
            models_selected.remove("Best")
        if "Gen+Segm" in models:
            models_selected += [m for m in MODELS_GAN_XNET]
            models_selected.remove("Gen+Segm")
        seen = set()
        models_selected = [x for x in models_selected if not (x in seen or seen.add(x))]
    if "All" in models:
        models_selected = None
    return models_selected


@app.callback(
    Output('loading-data-output', "children"),
    Output('df-total', "data"),
    Output('df-volume', "data"),
    Output('df-signature', "data"),
    Input('load_data', "n_clicks")
)
def load_data_spinner(n_clicks):
    testset = TestSet(DATA_PATH, load=True,
                      data_load=False, evaluation_load=False, radiomics_load=False)
    df_total = testset.df_total_evaluation
    df_volume = testset.df_volume_features
    df_signature = testset.df_signature_gt_3d
    return "Data is loaded.", df_total.to_json(), df_volume.to_json(), df_signature.to_json()


@app.callback(
    Output('theme', 'data'),
    Output('submit_theme', "style"),
    Input('submit_theme', "n_clicks"))
def switch_theme(n_clicks):
    if n_clicks == 0:
        raise PreventUpdate
    if n_clicks % 2 == 0:
        return json.dumps("gray"), {'color': 'white', 'backgroundColor': 'darkgray'}
    else:
        return json.dumps("plasma"), {'color': 'white', 'backgroundColor': 'purple'}


@app.callback(
    Output('heatmap-overview-slider', "step"),
    Output('heatmap-overview-slider', "max"),
    Output('heatmap-overview-slider', "value"),
    Input('dropdown-error-metric', "value"))
def update_slider_overview(metric):
    steps = {"DSC": 0.01,
             "ASSD": 1,
             "ACC": 0.01,
             "TPR": 0.01,
             "TNR": 0.01}
    max_values = {"DSC": 1,
                  "ASSD": MAX_VALUE_ASSD,
                  "ACC": 1,
                  "TPR": 1,
                  "TNR": 1}
    return steps[metric], max_values[metric], [0, max_values[metric]]


@app.callback(
    Output('heatmap-detail-slider', "step"),
    Output('heatmap-detail-slider', "max"),
    Output('heatmap-detail-slider', "value"),
    Input('dropdown-error-metric', "value"))
def update_slider_detail(metric):
    steps = {"DSC": 0.01,
             "ASSD": 1,
             "ACC": 0.01,
             "TPR": 0.01,
             "TNR": 0.01}
    max_values = {"DSC": 1,
                  "ASSD": MAX_VALUE_ASSD,
                  "ACC": 1,
                  "TPR": 1,
                  "TNR": 1}
    return steps[metric], max_values[metric], [0, max_values[metric]]


@app.callback(
    Output('json-selected-models', "data"),
    Output('dropdown-model', "value"),
    Input('submit-model', "n_clicks"),
    State('dropdown-model', "value"))
def update_selected_models(n_clicks, selected_models):
    if selected_models is None or len(selected_models) == 0:
        return json.dumps([]), []
    else:
        selected_models_processed = get_selected_model_list(selected_models)
        if selected_models_processed is None:
            selected_models_processed = MODELS
        return json.dumps(selected_models_processed), selected_models_processed


@app.callback(
    Output('df-metric-overview', "data"),
    Output('radioitem-dataset', "value"),
    Input('df-total', "data"),
    Input('json-selected-models', "data"),
    Input('dropdown-error-metric', "value"),
    Input('radioitem-dataset', "value"),
    Input("shortcut-mask", "n_keydowns"),
    State("shortcut-mask", "keydown"))
def update_data_overview(json_df_total, json_selected_models, selected_metric, selected_dataset, n_keys, keys):
    if json_df_total is None or json_selected_models is None:
        raise PreventUpdate
    else:
        if keys is not None:
            if keys['key'] == 'd':
                if selected_dataset == "only_tumor":
                    selected_dataset = "all"
                else:
                    selected_dataset = "only_tumor"
        df_total = pd.read_json(json_df_total)
        selected_models = json.loads(json_selected_models)
        if selected_models is not None and len(selected_models) == 0:
            return pd.DataFrame().to_json()
        lookup = {"DSC": f"dice_{selected_dataset}",
                  "ASSD": f"assd_{selected_dataset}",
                  "ACC": f"acc_{selected_dataset}",
                  "TPR": f"tpr_{selected_dataset}",
                  "TNR": f"tnr_{selected_dataset}"}
        df_metric = pd.DataFrame(df_total.iloc[0][lookup[selected_metric]])
        models_selected = ["id"] + selected_models if selected_models is not None else None
        if models_selected is not None:
            df_metric = df_metric[models_selected]
        return df_metric.to_json(), selected_dataset


# @app.callback(
#     Output('df-signature', "data"),
#     Output('radioitem-mask', "value"),
#     Input('radioitem-mask', "value"),
#     Input('shortcut-mask', "n_keydowns"),
#     State('shortcut-mask', "keydown")
# )
# def update_mask_signature(mask_mode, n_keys, keys):
#     testset = TestSet(DATA_PATH, load=True,
#                       data_load=False, evaluation_load=False, radiomics_load=False, full_mask=True)
#     if mask_mode == "margin_mask":
#         testset.mask_mode = False
#     if keys is not None:
#         if keys['key'] == 'm':
#             if mask_mode == "margin_mask":
#                 mask_mode = "full_mask"
#             else:
#                 mask_mode = "margin_mask"
#     df_signature = testset.df_signature_3d
#     return df_signature.to_json(), mask_mode


@app.callback(
    Output('df-heatmap-overview-overlay', "data"),
    Input('df-metric-overview', "data"),
    Input('df-feature-overview', "data"),
    Input('parcats-overview', "hoverData"))
def update_heatmap_overview_overlay(json_df_metric, json_df_features, selected_ids_hover):
    ctx = dash.callback_context
    if selected_ids_hover is not None and 'parcats-overview.hoverData' in ctx.triggered[0]["prop_id"]:
        df_metric = pd.read_json(json_df_metric)
        df_metric2 = df_metric.copy()
        df_features = pd.read_json(json_df_features)
        list_of_points = selected_ids_hover["points"]
        pointNumbers = []
        for p in list_of_points:
            pointNumbers.append(p["pointNumber"])
        selected_ids = [str(x) for x in df_features["id"][pointNumbers].values.tolist()]
        idx_selected = [int(idx) for idx, row in df_metric.iterrows() if row["id"] not in selected_ids][
                       :-1]
        idx_selected2 = [int(idx) for idx, row in df_metric.iterrows() if row["id"] in selected_ids] + [
            len(df_metric) - 1]
        df_metric2.iloc[idx_selected, 1:] = 1
        df_metric2.iloc[idx_selected2, 1:] = np.NaN
        return df_metric2.to_json()
    else:
        return pd.DataFrame().to_json()


@app.callback(
    Output('clickdata-parcats-overview', "data"),
    Input('parcats-overview', 'clickData'),
    Input('reset-button-overview', "n_clicks"))
def update_clickdata_parcats_overview(selected_ids, n_clicks):
    if n_clicks is None or selected_ids is None:
        return json.dumps(None)
    else:
        ctx = dash.callback_context
        if "reset-button-overview.n_clicks" in ctx.triggered[0]["prop_id"]:
            return json.dumps(None)
        elif selected_ids is not None:
            return json.dumps(selected_ids)


@app.callback(
    Output('heatmap-overview', "figure"),
    Input('df-metric-overview', "data"),
    Input('heatmap-overview-slider', "value"),
    Input('heatmap-overview-slider', "max"),
    Input('df-heatmap-overview-overlay', "data"),
    Input('df-feature-overview', "data"),
    Input('clickdata-parcats-overview', "data"),
    Input('reset-button-overview', "n_clicks"))
def update_heatmap_overview(json_df_metric, slider_values, slider_max, json_df_overlay, json_df_features, selected_ids,
                            n_clicks):
    if json_df_metric is None:
        raise PreventUpdate
    else:
        ctx = dash.callback_context
        selected_ids = json.loads(selected_ids)
        df_metric2 = pd.DataFrame()
        if json_df_overlay is not None:
            df_metric2 = pd.read_json(json_df_overlay)

        if "reset-button-overview.n_clicks" in ctx.triggered[0]["prop_id"]:
            data_selected = False
            data_hover = False
        elif "df-heatmap-overview-overlay.data" in ctx.triggered[0]["prop_id"] and len(df_metric2) != 0:
            data_selected = False
            data_hover = True
        elif selected_ids is not None:
            data_selected = True
            data_hover = False
        else:
            data_selected = False
            data_hover = False

        # read dataframe
        df_metric = pd.read_json(json_df_metric)
        if len(df_metric) == 0:
            return fig_no_model_selected
        metric = df_metric.iloc[-1]["id"]

        # define colorscale and tickvals
        colorscale, tickvals = get_colorscale_tickvals(metric, slider_values, slider_max)

        # filter for selected data points
        if data_selected:
            df_features = pd.read_json(json_df_features)
            list_of_points = selected_ids["points"]
            pointNumbers = []
            for p in list_of_points:
                pointNumbers.append(p["pointNumber"])
            selected_ids = [str(x) for x in df_features["id"][pointNumbers].values.tolist()]
            idx_selected = [int(idx) for idx, row in df_metric.iterrows() if row["id"] in selected_ids] + [
                len(df_metric) - 1]
            df_metric = df_metric.iloc[idx_selected, :]

        # create figure
        fig = make_subplots(rows=2, cols=1,
                            row_heights=[0.1, 0.9], vertical_spacing=0.05, shared_xaxes=True)
        # create annotated heatmap with total values
        round_dec = 1 if len(df_metric.columns) >= 15 else 2
        round_dec = round_dec if len(df_metric.columns) >= 11 else 3
        round_dec = round_dec if len(df_metric.columns) >= 7 else 4
        trace = ff.create_annotated_heatmap(x=list(df_metric.columns)[1:],
                                            y=["mean"],
                                            z=[list(df_metric.iloc[-1][1:].values)],
                                            hoverinfo='skip',
                                            coloraxis="coloraxis",
                                            annotation_text=[
                                                [np.round(x, round_dec) for x in
                                                 list(df_metric.iloc[-1][1:].values)]])
        fig.add_trace(trace.data[0], 1, 1)
        fig.layout.update(trace.layout)

        # prepare x,y,z for heatmap
        x = list(df_metric.columns)[1:]
        y = [str(x) for x in list(df_metric["id"].values[:-1])]
        z = [list(df_metric.iloc[idx][1:].values) for idx in range(len(df_metric) - 1)]
        # create hovertext
        hovertext = list()
        for yi, yy in enumerate(y):
            hovertext.append(list())
            for xi, xx in enumerate(x):
                hovertext[-1].append(
                    'Model: {}<br />ID: {}<br />{}: {}'.format(xx, yy, metric, np.round(z[yi][xi], decimals=5)))
        # heatmap for patient data
        fig.add_trace(go.Heatmap(x=x,
                                 y=y,
                                 z=z,
                                 zmin=0,
                                 zmax=1,
                                 zmid=1 / 2,
                                 zauto=False,
                                 hoverongaps=True,
                                 hoverinfo='text',
                                 text=hovertext,
                                 coloraxis="coloraxis"), 2, 1)
        # if selected ids -> highlight only those
        if data_hover:
            colorscale_nan = px.colors.colorbrewer.Greys[0:2]  # px.colors.colorbrewer.Greys_r
            x2 = list(df_metric2.columns[1:])
            y2 = [str(x) for x in list(df_metric2["id"].values[:-1])]
            z2 = [list(df_metric2.iloc[idx][1:].values) for idx in range(len(df_metric2) - 1)]
            fig.add_trace(go.Heatmap(
                x=x2,
                y=y2,
                z=z2,
                hoverongaps=False,
                hoverinfo='skip',
                colorscale=colorscale_nan,
                showscale=False), 2, 1)
        # update layout
        fig.update_layout(xaxis2={'showticklabels': False},
                          xaxis1={'side': 'top', 'showticklabels': True},
                          yaxis2={'title': 'Patient ID'},
                          coloraxis={'colorscale': colorscale, 'cmin': tickvals[0], 'cmax': tickvals[-1],
                                     'colorbar': dict(title=metric, tickvals=tickvals, tickmode="array")},
                          margin=dict(l=5, r=5, b=5, t=5, pad=4))
    return fig


@app.callback(
    Output('dict-patient-id', "data"),
    Input('heatmap-overview', "clickData"))
def update_patient_id_detail(clickData):
    if clickData is None:
        raise PreventUpdate
    else:
        patient_id = clickData["points"][0]["y"]
        return json.dumps(patient_id)


@app.callback(
    Output('dict-slice-id', "data"),
    Input('heatmap-detail', "clickData"))
def update_slice_id_detail(clickData):
    if clickData is None:
        raise PreventUpdate
    else:
        slice_id = clickData["points"][0]["y"]
        return json.dumps(slice_id)


@app.callback(
    Output('slice-slider', "value"),
    Output('header-slice', "children"),
    Output('slice-slider', "min"),
    Output('slice-slider', "max"),
    Output('slice-slider', "marks"),
    Input('df-metric-detail', "data"),
    Input('dict-slice-id', "data"),
    Input('slice-slider', "value"))
def update_slice_slider(df_metric_json, json_slice_id, current_slice_id):
    ctx = dash.callback_context
    if df_metric_json is None:
        raise PreventUpdate
    else:
        df = pd.read_json(df_metric_json)
        if len(df) == 0:
            raise PreventUpdate
        min_slice = df.iloc[0]["slice"]
        max_slice = df.iloc[-2]["slice"]
        if max_slice - min_slice < 20:
            marks = {k: {'label': str(k)} for k in range(min_slice, max_slice + 1, 1)}
        else:
            marks = {k: {'label': str(k)} if idx % 5 == 0 else {'label': ""} for idx, k in
                     enumerate(range(min_slice, max_slice + 1, 1))}
            marks[max_slice] = {'label': str(max_slice)}
        if json_slice_id is None:
            current_slice = int(min_slice + (max_slice - min_slice) / 2)
            return current_slice, f"Slice ID {current_slice}", min_slice, max_slice, marks
        if "dict-slice-id.data" == ctx.triggered[0]["prop_id"]:
            current_slice = int(json.loads(json_slice_id))
        else:
            current_slice = current_slice_id
        return current_slice, f"Slice ID {current_slice}", min_slice, max_slice, marks


@app.callback(
    Output('df-metric-detail', "data"),
    Input('dict-patient-id', "data"),
    Input('json-selected-models', "data"),
    Input('dropdown-error-metric', "value"),
    Input('radioitem-dataset', "value"))
def update_data_detail(json_patient_id, json_selected_models, selected_metric, selected_dataset):
    if json_patient_id is None:
        raise PreventUpdate
    else:
        patient_id = int(json.loads(json_patient_id))
        df = pd.read_json(f"{DATA_PATH}/vs_gk_{patient_id}/evaluation.json")
        selected_models = json.loads(json_selected_models)
        if selected_models is not None and len(selected_models) == 0:
            return pd.DataFrame().to_json()
        if selected_metric in ["DSC", "ASSD"]:
            lookup = {"DSC": "dice", "ASSD": "assd"}
            cols = [c for c in df.columns if lookup[selected_metric] in c]
            df_metric = df[["slice", "VS_class_gt"] + cols]
            df_metric.rename(columns={k: k.split("-")[-1] for k in cols}, inplace=True)
            models_selected = ["slice", "VS_class_gt"] + selected_models if selected_models is not None else None
            if models_selected is not None:
                df_metric = df_metric[models_selected]
            if selected_dataset == "only_tumor":
                df_metric = df_metric[df_metric["VS_class_gt"] == 1]
            df_metric.drop(columns=["VS_class_gt"], inplace=True)
            df_metric = df_metric.append({"slice": selected_metric, **dict(df_metric.mean()[1:])}, ignore_index=True)

        else:  # ["ACC", "TPR", "TNR"]
            cols = [c for c in df.columns if "class_pred-" in c]
            df_metric = df[["slice", "VS_class_gt"] + cols]
            df_metric.rename(columns={k: k.split("-")[-1] for k in cols}, inplace=True)
            models_selected = ["slice", "VS_class_gt"] + selected_models if selected_models is not None else None
            if models_selected is not None:
                df_metric = df_metric[models_selected]
            if selected_dataset == "only_tumor":
                df_metric = df_metric[df_metric["VS_class_gt"] == 1]
            lookup = {"ACC": TestSet().calculate_accuracy,
                      "TPR": TestSet().calculate_tpr,
                      "TNR": TestSet().calculate_tnr}
            model_cols = list(df_metric.columns)[2:]
            values = [lookup[selected_metric](
                confusion_matrix(df_metric["VS_class_gt"].values, x[1].values, labels=[0, 1]).ravel()) for x in
                df_metric[model_cols].items()]
            df_metric.drop(columns=["VS_class_gt"], inplace=True)
            df_metric = df_metric.append(
                {"slice": selected_metric, **{k: v for k, v in zip(model_cols, values)}}, ignore_index=True)

        return df_metric.to_json()


@app.callback(
    Output('df-heatmap-detail-overlay', "data"),
    Input('df-metric-detail', "data"),
    Input('df-feature-detail', "data"),
    Input('parcats-detail', "hoverData"),
    Input('slice-slider', "value")
)
def update_heatmap_detail_overlay(json_df_metric, json_df_features, selected_ids_hover, current_slice):
    ctx = dash.callback_context
    if selected_ids_hover is not None and 'parcats-detail.hoverData' in ctx.triggered[0]["prop_id"]:
        df_metric = pd.read_json(json_df_metric)
        df_metric2 = df_metric.copy()
        df_features = pd.read_json(json_df_features)
        list_of_points = selected_ids_hover["points"]
        pointNumbers = []
        for p in list_of_points:
            pointNumbers.append(p["pointNumber"])
        if "id" in df_features.columns:
            selected_ids = df_features["id"][pointNumbers].values.tolist()
            idx_selected = [int(idx) for idx, row in df_metric.iterrows() if row["slice"] not in selected_ids][:-1]
            idx_selected2 = [int(idx) for idx, row in df_metric.iterrows() if row["slice"] in selected_ids] + [
                len(df_metric) - 1]
            df_metric2.iloc[idx_selected, 1:] = 1
            df_metric2.iloc[idx_selected2, 1:] = np.NaN
        elif "model" in df_features.columns:
            selected_ids = df_features["model"][pointNumbers].values.tolist()
            idx_selected = [c for c in df_metric.columns if c not in selected_ids + ["slice"]]
            idx_selected2 = [c for c in df_metric.columns if c in selected_ids]
            idx = np.where(df_metric["slice"].values == current_slice)[0][0]
            df_metric2[idx_selected2] = 1
            df_metric2.iloc[idx, 1:] = np.NaN
            df_metric2[idx_selected] = 1
        return df_metric2.to_json()
    else:
        pd.DataFrame().to_json()


@app.callback(
    Output('clickdata-parcats-detail', "data"),
    Input('parcats-detail', 'clickData'),
    Input('reset-button-detail', "n_clicks"))
def update_clickdata_parcats_detail(selected_ids, n_clicks):
    if n_clicks is None or selected_ids is None:
        return json.dumps(None)
    else:
        ctx = dash.callback_context
        if "reset-button-detail.n_clicks" in ctx.triggered[0]["prop_id"]:
            return json.dumps(None)
        elif selected_ids is not None:
            return json.dumps(selected_ids)


@app.callback(
    Output('heatmap-detail', "figure"),
    Output('header-detail', "children"),
    Input('df-metric-detail', "data"),
    Input('heatmap-detail-slider', "value"),
    Input('heatmap-detail-slider', "max"),
    Input('df-heatmap-detail-overlay', "data"),
    Input('df-feature-detail', "data"),
    Input('clickdata-parcats-detail', "data"),
    Input('reset-button-detail', "n_clicks"),
    Input('dict-patient-id', "data"))
def update_heatmap_detail(df_metric_json, slider_values, slider_max, json_df_overlay, json_df_features, selected_ids,
                          n_clicks, json_patient_id):
    if df_metric_json is None:
        raise PreventUpdate
    else:
        patient_id = int(json.loads(json_patient_id))
        ctx = dash.callback_context
        selected_ids = json.loads(selected_ids)
        df_metric2 = pd.DataFrame()
        if json_df_overlay is not None:
            df_metric2 = pd.read_json(json_df_overlay)

        if "reset-button-detail.n_clicks" in ctx.triggered[0]["prop_id"] or "dict-patient-id.data" in ctx.triggered[0][
            "prop_id"]:
            data_selected = False
            data_hover = False
        elif "df-heatmap-detail-overlay.data" in ctx.triggered[0]["prop_id"] and len(df_metric2) != 0:
            data_selected = False
            data_hover = True
        elif selected_ids is not None:
            data_selected = True
            data_hover = False
        else:
            data_selected = False
            data_hover = False

        # read dataframe
        df_metric = pd.read_json(df_metric_json)
        if len(df_metric) == 0:
            return fig_no_model_selected
        metric = df_metric.iloc[-1]["slice"]

        # define colorscale and tickvals
        colorscale, tickvals = get_colorscale_tickvals(metric, slider_values, slider_max)

        # filter for selected data points
        if data_selected:
            df_features = pd.read_json(json_df_features)
            list_of_points = selected_ids["points"]
            pointNumbers = []
            for p in list_of_points:
                pointNumbers.append(p["pointNumber"])
            if "id" in df_features.columns:
                selected_ids = [str(x) for x in df_features["id"][pointNumbers].values.tolist()]
                idx_selected = [int(idx) for idx, row in df_metric.iterrows() if str(row["slice"]) in selected_ids] + [
                    len(df_metric) - 1]
                df_metric = df_metric.iloc[idx_selected, :]
            if "model" in df_features.columns:
                selected_ids = ["slice"] + df_features["model"][pointNumbers].values.tolist()
                df_metric = df_metric[selected_ids]

        # create figure
        fig = make_subplots(rows=2, cols=1,
                            row_heights=[0.1, 0.9], vertical_spacing=0.05, shared_xaxes=True)
        # create annotated heatmap with total values
        round_dec = 1 if len(df_metric.columns) >= 15 else 2
        round_dec = round_dec if len(df_metric.columns) >= 11 else 3
        round_dec = round_dec if len(df_metric.columns) >= 7 else 4
        trace = ff.create_annotated_heatmap(x=list(df_metric.columns)[1:],
                                            y=["mean"],
                                            z=[list(df_metric.iloc[-1][1:].values)],
                                            hoverinfo='skip',
                                            coloraxis="coloraxis",
                                            annotation_text=[
                                                [np.round(x, round_dec) for x in list(df_metric.iloc[-1][1:].values)]])
        fig.add_trace(trace.data[0], 1, 1)
        fig.layout.update(trace.layout)

        # prepare x,y,z for heatmap
        x = list(df_metric.columns)[1:]
        y = [str(x) for x in list(df_metric["slice"].values[:-1])]
        z = [list(df_metric.iloc[idx][1:].values) for idx in range(len(df_metric) - 1)]
        # create hovertext
        hovertext = list()
        for yi, yy in enumerate(y):
            hovertext.append(list())
            for xi, xx in enumerate(x):
                hovertext[-1].append(
                    'Model: {}<br />Slice: {}<br />{}: {}'.format(xx, yy, metric, np.round(z[yi][xi], decimals=5)))
        # heatmap for patient data
        fig.add_trace(go.Heatmap(x=x,
                                 y=y,
                                 z=z,
                                 hoverongaps=False,
                                 hoverinfo='text',
                                 text=hovertext,
                                 coloraxis="coloraxis"), 2, 1)
        # if selected ids -> highlight only those
        if data_hover:
            colorscale_nan = px.colors.colorbrewer.Greys[0:2]
            x2 = list(df_metric2.columns[1:])
            y2 = [str(x) for x in list(df_metric2["slice"].values[:-1])]
            z2 = [list(df_metric2.iloc[idx][1:].values) for idx in range(len(df_metric2) - 1)]
            fig.add_trace(go.Heatmap(
                x=x2,
                y=y2,
                z=z2,
                hoverongaps=False,
                hoverinfo='skip',
                colorscale=colorscale_nan,
                showscale=False), 2, 1)
        # update layout
        fig.update_layout(xaxis2={'showticklabels': False},
                          xaxis1={'side': 'top', 'showticklabels': True},
                          yaxis2={'title': 'Slice'},
                          coloraxis={'colorscale': colorscale, 'cmin': tickvals[0], 'cmax': tickvals[-1],
                                     'colorbar': dict(title=metric, tickvals=tickvals, tickmode="array")},
                          margin=dict(l=5, r=5, b=5, t=5, pad=4))
        return fig, f"Patient ID {patient_id}"


@app.callback(
    Output('dict-slice-data', "data"),
    Input('dict-patient-id', "data"),
    Input('slice-slider', "value"),
    Input('json-selected-models', "data"),
    Input('df-feature-detail', "data"),
    Input('clickdata-parcats-detail', "data")
)
def update_data_slice(json_patient_id, slice_id, json_selected_models, json_df_features, selected_ids):
    if json_patient_id is None or slice_id is None or json_selected_models is None or json_df_features is None:
        raise PreventUpdate
    else:
        patient_id = int(json.loads(json_patient_id))
        slice = slice_id

        selected_ids = json.loads(selected_ids)
        df_features = pd.read_json(json_df_features)
        if selected_ids is not None and df_features is not None and "model" in df_features.keys():
            list_of_points = selected_ids["points"]
            pointNumbers = []
            for p in list_of_points:
                pointNumbers.append(p["pointNumber"])
            selected_models = df_features["model"][pointNumbers].values.tolist()
        else:
            selected_models = json.loads(json_selected_models)
            if selected_models is not None and len(selected_models) == 0:
                return json.dumps({})
            if selected_models is None:
                selected_models = MODELS

        df = pd.read_json(f"{DATA_PATH}/vs_gk_{patient_id}/evaluation.json")
        container = DataContainer(f"{DATA_PATH}/vs_gk_{patient_id}/")
        img = container.t2_scan_slice(slice)
        segmentation_filled = []
        for m in selected_models:
            segm_filled = cv2.drawContours(np.zeros((256, 256)),
                                           [np.array(s).astype(np.int64) for s in
                                            np.array(df.iloc[slice][f"VS_segm_pred-{m}"], dtype="object")], -1, (1),
                                           -1)
            segmentation_filled.append(segm_filled)
        segm_sum = (np.sum(segmentation_filled, axis=0))
        z_max = len(segmentation_filled)
        gt_filled = cv2.drawContours(np.zeros((256, 256)),
                                     [np.array(s).astype(np.int64) for s in
                                      np.array(df.iloc[slice][f"VS_segm_gt"], dtype="object")], -1,
                                     (len(segmentation_filled)),
                                     -1)
        gt_contour = cv2.drawContours(np.zeros((256, 256)),
                                      [np.array(s).astype(np.int64) for s in
                                       np.array(df.iloc[slice][f"VS_segm_gt"], dtype="object")], -1,
                                      (len(segmentation_filled)), 1)
        segm_subtract = gt_filled - segm_sum
        info_dict = {"slice": slice,
                     "patient_id": patient_id,
                     "selected_models": selected_models,
                     "z_max": z_max,
                     "img": img.tolist(),
                     "segm_sum": segm_sum.tolist(),
                     "segm_subtract": segm_subtract.tolist(),
                     "gt_contour": gt_contour.tolist(),
                     "gt_filled": gt_filled.tolist()}
        return json.dumps(info_dict)


@app.callback(
    Output('control_gt_2', "style"),
    Input('radioitem-gt-toggle', "value"))
def update_gt_control(gt_toggle):
    if gt_toggle == "show":
        return {'display': 'table', 'width': '97%'}
    else:
        return {'display': 'none', 'width': '97%'}


@app.callback(
    Output('slice-plot', "figure"),
    Input('dict-slice-data', "data"),
    Input('radioitem-slice-view', "value"),
    Input('radioitem-gt-toggle', "value"),
    Input('radioitem-gt-type', "value"),
    Input('slider-mask-opacity', "value"),
    Input('slider-gt-opacity', "value"))
def update_heatmap_slice(json_dict_slice_data, view_type, gt_toggle, gt_type, mask_opacity, gt_opacity):
    if json_dict_slice_data is None:
        raise PreventUpdate
    else:
        info_dict = json.loads(json_dict_slice_data)
        if len(info_dict) == 0:
            return fig_no_model_selected
        # image
        img = np.array(info_dict["img"])
        # gt
        gt = np.zeros_like(img)
        if gt_toggle == "show":
            if gt_type == "contour":
                gt = np.array(info_dict["gt_contour"])
            else:
                gt = np.array(info_dict["gt_filled"])
        # selected models
        z_max = int(info_dict["z_max"])
        selected_models = info_dict["selected_models"]
        if view_type == "sum":
            # select segm
            segm = np.array(info_dict["segm_sum"])
            # discrete colorscale
            bvals = list(range(0, z_max + 1, 1))
            nvals = [(v - bvals[0]) / (bvals[-1] - bvals[0]) for v in bvals]
            if z_max > 1:
                colors = n_colors('rgba(246, 192, 174)', 'rgb(174, 57, 18)', z_max, colortype='rgb')
            else:
                colors = ['rgb(174, 57, 18)']
            colors = [c.replace("rgb", "rgba").replace(")", ",1)") if "rgba" not in c else c for c in colors]
        else:
            # select segm
            segm = np.array(info_dict["segm_subtract"])
            # discrete colorscale
            bvals = list(range(-z_max - 1, z_max + 1, 1))
            nvals = [(v - bvals[0]) / (bvals[-1] - bvals[0]) for v in bvals]
            if z_max > 1:
                red = n_colors('rgba(246, 192, 174)', 'rgb(174, 57, 18)', z_max, colortype='rgb')
                blue = n_colors('rgb(70, 3, 159)', 'rgb(231, 213, 254)', z_max, colortype='rgb')
            else:
                red = ['rgb(174, 57, 18)']
                blue = ['rgb(70, 3, 159)']
            z_max = z_max * 2
            colors = blue + ['rgba(255,255,255,1)'] + red
            colors = [c.replace("rgb", "rgba").replace(")", ",1)") if "rgba" not in c else c for c in colors]

        # colorscale gt
        colorscale_gt = [[0, 'rgba(0,0,0,0)'],
                         [0.99, 'rgba(0,0,0,0)'],
                         [1, 'rgba(0,0,0,1)']]

        # discrete colorscale segm
        dcolorscale = []
        for k in range(len(colors)):
            dcolorscale.extend([[nvals[k], colors[k]], [nvals[k + 1], colors[k]]])
        # colorbar ticks
        bvals2 = np.array(bvals)
        tickvals = np.linspace(0.5, z_max - 0.5, len(colors)).tolist()
        ticktext = [f"{k}" for k in bvals2[1:]]

        # hovertext
        segm[segm == 0] = None
        hovertext = list()
        for yi, yy in enumerate(segm):
            hovertext.append(list())
            for xi, xx in enumerate(segm[0]):
                hovertext[-1].append('value: {}'.format(segm[yi][xi]))

        # figure
        fig = go.Figure()
        fig_img = px.imshow(img, binary_string=True)
        fig.add_trace(fig_img.data[0])
        fig.update_traces(hovertemplate=None, hoverinfo="skip")
        fig.update_traces(opacity=1.0)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)

        if view_type == "subtraction":
            segm[segm != 0] += z_max // 2
        else:
            segm[segm != 0] -= 1
        fig.add_heatmap(z=segm, showscale=True, colorscale=dcolorscale,
                        zmin=0, zmax=z_max, hoverongaps=False, text=hovertext, hoverinfo='text',
                        colorbar=dict(thickness=30, tickmode="array", tickvals=tickvals, ticktext=ticktext),
                        opacity=mask_opacity, name="segm")

        fig.add_heatmap(z=gt, hoverinfo="skip", showscale=False, colorscale=colorscale_gt,
                        opacity=gt_opacity, name="gt")

        fig.update_layout(margin=dict(l=0, r=0, b=5, t=15, pad=4), uirevision=True)
        return fig


@app.callback(
    Output('df-feature-overview', "data"),
    Output('dict-bin-mapping', "data"),
    Input('df-total', "data"),
    Input('df-signature', "data"),
    Input('df-volume', "data"),
    Input('json-selected-models', "data"))
def update_data_parcats_overview(df_total_json, df_signature_json, df_volume_json, json_selected_models):
    if df_total_json is None or df_signature_json is None or df_volume_json is None:
        raise PreventUpdate
    else:
        # load data
        df_total = pd.read_json(df_total_json)
        df_signature = pd.read_json(df_signature_json)
        df_volume = pd.read_json(df_volume_json)
        selected_models = json.loads(json_selected_models)
        if selected_models is not None and len(selected_models) == 0:
            return pd.DataFrame().to_json()
        models_selected = ["id"] + selected_models if selected_models is not None else None
        # performance features
        df_performance = pd.DataFrame()
        df_performance["id"] = list(df_total["dice_all"][0]["id"].values())[:-1]
        values = [3, 2, 1]
        values_ascending = dict(enumerate(values, 1))
        values_descending = dict(enumerate(reversed(values), 1))
        dict_performance_bins = {}
        for m in ["dice", "assd", "acc", "tnr", "tpr"]:
            for d in ["all", "only_tumor"]:
                df_metric = pd.DataFrame(df_total[m + "_" + d][0])[models_selected]
                df_metric["mean_value"] = df_metric.mean(axis=1)
                bins = np.linspace(np.min(df_metric["mean_value"]),
                                   np.nextafter(np.max(df_metric["mean_value"]), np.inf), 4)
                if "assd" in m:
                    df_metric["mean"] = np.vectorize(values_descending.get)(
                        np.digitize(df_metric["mean_value"], bins=bins))
                else:
                    df_metric["mean"] = np.vectorize(values_ascending.get)(
                        np.digitize(df_metric["mean_value"], bins=bins))
                df_performance[m + "_" + d] = df_metric.iloc[:-1]["mean"].values
                if m + "_" + d not in dict_performance_bins.keys():
                    dict_performance_bins[m + "_" + d] = bins.tolist()
        # radiomics features
        df_radiomics = df_signature[
            ["id"] + [c for c in df_signature.columns if c.split("-")[0] in ["shape", "firstorder"]]]
        df_radiomics["id"] = df_radiomics["id"].apply(lambda x: str(x))
        featureKeys = list(df_radiomics.keys())
        featureKeys.remove("id")
        # volumetric features
        df_volume["id"] = df_volume["id"].apply(lambda x: str(x))
        bins = [0, 70, 79, 80]
        df_volume["slice_number"] = np.digitize(df_volume["slice_number"], bins=bins)
        bins = [0, 10, 15, 20]
        df_volume["tumor_slice_number"] = np.digitize(df_volume["tumor_slice_number"], bins=bins)
        # merge features
        df_merged = df_volume.merge(df_performance.merge(df_radiomics, on="id"), on="id")
        df_merged = df_merged.fillna(value=0)
        return df_merged.to_json(), json.dumps(dict_performance_bins)


@app.callback(
    Output("dropdown-pcd-label", "style"),
    Output("dropdown-pcd-type", "style"),
    Input("dropdown-pcd-cluster", "value")
)
def update_pcd_type(pcd_type):
    if pcd_type is None:
        raise PreventUpdate
    else:
        if pcd_type == "slice_pcd":
            style1 = {'width': '15%',
                      'textAlign': 'left',
                      'paddingLeft': '1%',
                      'verticalAlign': 'middle',
                      'display': 'none'}
            style2 = {'width': '15%',  # TODO adapt width
                      'paddingRight': '1%',
                      'verticalAlign': 'middle',
                      'display': 'none'}
            return style1, style2
        elif pcd_type == "model_pcd":
            style1 = {'width': '15%',
                      'textAlign': 'left',
                      'paddingLeft': '1%',
                      'verticalAlign': 'middle',
                      'display': 'table-cell'}
            style2 = {'width': '15%',  # TODO adapt width
                      'paddingRight': '1%',
                      'verticalAlign': 'middle',
                      'display': 'table-cell'}
            return style1, style2


def get_feature_lookup_table(features, metric, dataset, type_level):
    feature_list = []
    if features == "firstorder":
        feature_list = ["Energy", "Skewness", "Kurtosis", "Variance", "Range", "MeanAbsoluteDeviation"]
        # "Entropy", "Uniformity"
    elif features == "shape":
        if type_level == "overview":
            feature_list = ["MeshVolume", "SurfaceArea", "Sphericity", "Elongation", "Flatness", "Maximum3DDiameter"]
        else:
            feature_list = ["MeshSurface", "Sphericity", "Elongation", "MaximumDiameter"]
        # "VoxelVolume"
    elif features == "performance":
        if type_level == "overview":
            feature_list = ['dice_all', 'dice_only_tumor', 'assd_all', 'assd_only_tumor',
                            'acc_all', 'acc_only_tumor', 'tnr_all', 'tpr_all']
        else:
            feature_list = ['dice_all', 'dice_only_tumor', 'assd_all', 'assd_only_tumor']
        feature_list.remove(str(metric + "_" + dataset))
    return feature_list


def get_ticktext_performance(bin_mapping, f):
    if "assd" in f:
        vals = [np.round(x, 0) for x in bin_mapping[f]]
        bvals = [f"{vals[i]}-{vals[i + 1]}" for i in range(len(vals) - 1)]
        return list(([f"{a}<br>({b})" for a, b in zip(["good", "medium", "bad"], bvals)]))
    else:
        vals = [np.round(x, 2) for x in bin_mapping[f]]
        bvals = [f"{vals[i]}-{vals[i + 1]}" for i in range(len(vals) - 1)]
        return list(reversed([f"{a}<br>({b})" for a, b in zip(["bad", "medium", "good"], bvals)]))


def get_ticktext_size(bin_mapping, f):
    vals = [int(np.round(x)) for x in bin_mapping[f]]
    bvals = list(reversed([f"{vals[i]}-{vals[i + 1]}" for i in range(len(vals) - 1)]))
    return list(([f"{a}<br>({b})" for a, b in zip(["large", "medium", "small"], bvals)])) + ["absent"]


def get_lookup_dict_overview(bin_mapping):
    return {
        # firstorder
        "Energy": dict(ticktext=['large', 'medium', 'small'], categoryarray=[3, 2, 1]),
        "Variance": dict(ticktext=['large', 'medium', 'small'], categoryarray=[3, 2, 1]),
        "Skewness": dict(ticktext=['pos', 'neg'], categoryarray=[2, 1]),
        "Kurtosis": dict(ticktext=['lepto', 'platy'], categoryarray=[2, 1]),
        "Range": dict(ticktext=['large', 'medium', 'small'], categoryarray=[3, 2, 1]),
        "MeanAbsoluteDeviation": dict(ticktext=['large', 'medium', 'small'], categoryarray=[3, 2, 1],
                                      label="MAD"),
        # shape3D
        "MeshVolume": dict(ticktext=['large', 'medium', 'small'], categoryarray=[3, 2, 1]),
        "SurfaceArea": dict(ticktext=['large', 'medium', 'small'], categoryarray=[3, 2, 1]),
        "Sphericity": dict(ticktext=['>mean', '<=mean'], categoryarray=[2, 1]),
        "Elongation": dict(ticktext=['>mean', '<=mean'], categoryarray=[2, 1]),
        "Flatness": dict(ticktext=['non-flat', 'flat'], categoryarray=[2, 1]),
        "Maximum3DDiameter": dict(ticktext=['large', 'medium', 'small'], categoryarray=[3, 2, 1],
                                  label="Max3DDiam"),
        # performance
        "dice_only_tumor": dict(ticktext=get_ticktext_performance(bin_mapping, "dice_only_tumor"),
                                categoryarray=[1, 2, 3],
                                label="DSC tumor"),
        "dice_all": dict(ticktext=get_ticktext_performance(bin_mapping, "dice_all"),
                         categoryarray=[1, 2, 3],
                         label="DSC"),
        "acc_only_tumor": dict(ticktext=get_ticktext_performance(bin_mapping, "acc_only_tumor"),
                               categoryarray=[1, 2, 3],
                               label="ACC tumor"),
        "acc_all": dict(ticktext=get_ticktext_performance(bin_mapping, "acc_all"),
                        categoryarray=[1, 2, 3],
                        label="ACC"),
        "assd_only_tumor": dict(ticktext=get_ticktext_performance(bin_mapping, "assd_only_tumor"),
                                categoryarray=[1, 2, 3],
                                label="ASSD tumor"),
        "assd_all": dict(ticktext=get_ticktext_performance(bin_mapping, "assd_all"),
                         categoryarray=[1, 2, 3],
                         label="ASSD"),
        "tpr_only_tumor": dict(ticktext=get_ticktext_performance(bin_mapping, "tpr_only_tumor"),
                               categoryarray=[1, 2, 3],
                               label="TPR tumor"),
        "tpr_all": dict(ticktext=get_ticktext_performance(bin_mapping, "tpr_all"),
                        categoryarray=[1, 2, 3],
                        label="TPR"),
        "tnr_all": dict(ticktext=get_ticktext_performance(bin_mapping, "tnr_all"),
                        categoryarray=[1, 2, 3],
                        label="TNR"),
        # dataset 3D
        "slice_number": dict(ticktext=["80", "79", "70-78", "<70"], categoryarray=[4, 3, 2, 1], label="#Slices"),
        "tumor_slice_number": dict(ticktext=[">20", "15-20", "10-15", "<10"], categoryarray=[4, 3, 2, 1],
                                   label="#Tumor Slices")}


def get_lookup_dict_detail(bin_mapping, bin_mapping_size):
    return {
        # firstorder
        "Energy": dict(ticktext=['large', 'medium', 'small', 'absent'], categoryarray=[3, 2, 1, 0]),
        "Variance": dict(ticktext=['large', 'medium', 'small', 'absent'], categoryarray=[3, 2, 1, 0]),
        "Skewness": dict(ticktext=['pos', 'neg', 'absent'], categoryarray=[2, 1, 0]),
        "Kurtosis": dict(ticktext=['lepto', 'platy', 'absent'], categoryarray=[2, 1, 0]),
        "Range": dict(ticktext=['large', 'medium', 'small', 'absent'], categoryarray=[3, 2, 1, 0]),
        "MeanAbsoluteDeviation": dict(ticktext=['large', 'medium', 'small', 'absent'], categoryarray=[3, 2, 1, 0],
                                      label="MAD"),
        # shape2D
        "MeshSurface": dict(ticktext=['large', 'medium', 'small', 'absent'], categoryarray=[3, 2, 1, 0]),
        "Sphericity": dict(ticktext=['>mean', '<=mean', 'absent'], categoryarray=[2, 1, 0]),
        "Elongation": dict(ticktext=['>mean', '<=mean', 'absent'], categoryarray=[2, 1, 0]),
        "MaximumDiameter": dict(ticktext=['large', 'medium', 'small', 'absent'], categoryarray=[3, 2, 1, 0],
                                label="Max2DDiam"),
        # performance
        "dice_only_tumor": dict(ticktext=get_ticktext_performance(bin_mapping, "dice_only_tumor"),
                                categoryarray=[1, 2, 3],
                                label="DSC tumor"),
        "dice_all": dict(ticktext=get_ticktext_performance(bin_mapping, "dice_all"),
                         categoryarray=[1, 2, 3],
                         label="DSC"),
        "assd_only_tumor": dict(ticktext=get_ticktext_performance(bin_mapping, "assd_only_tumor"),
                                categoryarray=[1, 2, 3],
                                label="ASSD tumor"),
        "assd_all": dict(ticktext=get_ticktext_performance(bin_mapping, "assd_all"),
                         categoryarray=[1, 2, 3],
                         label="ASSD"),
        # dataset 2D
        "tumor_presence": dict(ticktext=['present', 'absent'], categoryarray=[2, 1], label="Presence"),
        "tumor_size": dict(ticktext=get_ticktext_size(bin_mapping_size, "tumor_size"), categoryarray=[3, 2, 1, 0],
                           label="Size(px)")}


def get_lookup_dict_detail_2():
    return {
        # firstorder
        "Energy": dict(ticktext=['large', 'medium', 'small', 'absent'], categoryarray=[3, 2, 1, 0]),
        "Variance": dict(ticktext=['large', 'medium', 'small', 'absent'], categoryarray=[3, 2, 1, 0]),
        "Skewness": dict(ticktext=['pos', 'neg', 'absent'], categoryarray=[2, 1, 0]),
        "Kurtosis": dict(ticktext=['lepto', 'platy', 'absent'], categoryarray=[2, 1, 0]),
        "Range": dict(ticktext=['large', 'medium', 'small', 'absent'], categoryarray=[3, 2, 1, 0]),
        "MeanAbsoluteDeviation": dict(ticktext=['large', 'medium', 'small', 'absent'], categoryarray=[3, 2, 1, 0],
                                      label="MAD"),
        # shape2D
        "MeshSurface": dict(ticktext=['large', 'medium', 'small', 'absent'], categoryarray=[3, 2, 1, 0]),
        "Sphericity": dict(ticktext=['>mean', '<=mean', 'absent'], categoryarray=[2, 1, 0]),
        "Elongation": dict(ticktext=['>mean', '<=mean', 'absent'], categoryarray=[2, 1, 0]),
        "MaximumDiameter": dict(ticktext=['large', 'medium', 'small', 'absent'], categoryarray=[3, 2, 1, 0],
                                label="Max2DDiam")
    }


@app.callback(
    Output('parcats-overview', "figure"),
    Input('df-feature-overview', "data"),
    Input('dict-bin-mapping', "data"),
    Input('dropdown-error-metric', "value"),
    Input('radioitem-dataset', "value"),
    Input('dropdown-feature', "value"),
    Input('theme', 'data')
)
def update_parcats_overview(df_plot_json, dict_bin_mapping, selected_metric, selected_dataset, selected_features,
                            theme):
    if df_plot_json is None or selected_metric is None or selected_dataset is None:
        raise PreventUpdate
    else:
        df_new = pd.read_json(df_plot_json)
        bin_mapping = json.loads(dict_bin_mapping)
        if len(df_new) == 0:
            return fig_no_model_selected
        # define plot df
        lookup = {"DSC": "dice",
                  "ASSD": "assd",
                  "ACC": "acc",
                  "TPR": "tpr",
                  "TNR": "tnr"}
        metric = lookup[selected_metric]
        feature_list = get_feature_lookup_table(selected_features, metric, selected_dataset, type_level="overview")
        df_plot = pd.DataFrame()
        df_plot["id"] = df_new["id"]
        df_plot["slice_number"] = df_new["slice_number"]
        df_plot["tumor_slice_number"] = df_new["tumor_slice_number"]
        performance_col = metric + "_" + selected_dataset
        df_plot[performance_col] = df_new[performance_col].values
        for feature in feature_list:
            df_plot[feature] = df_new[[c for c in df_new.columns if feature in c][0]].values
        plot_cols = df_plot.columns.to_list()
        plot_cols.remove("id")
        # Create dimensions
        lookup_dict = get_lookup_dict_overview(bin_mapping)
        perf_dim = [go.parcats.Dimension(values=df_plot[performance_col], **lookup_dict[performance_col]),
                    go.parcats.Dimension(values=df_plot["slice_number"], **lookup_dict["slice_number"]),
                    go.parcats.Dimension(values=df_plot["tumor_slice_number"], **lookup_dict["tumor_slice_number"]),
                    ]
        plot_cols.remove("slice_number")
        plot_cols.remove("tumor_slice_number")
        plot_cols.remove(performance_col)
        feature_dim = []
        for f in plot_cols:
            if "label" in lookup_dict[f].keys():
                feature_dim.append(go.parcats.Dimension(values=df_plot[f], **lookup_dict[f]))
            else:
                feature_dim.append(go.parcats.Dimension(values=df_plot[f], label=f, **lookup_dict[f]))
        # Create parcats trace
        color = df_plot[performance_col]
        if json.loads(theme) == "gray":
            colorscale = [[0, 'rgb(42,42,42)'], [0.5, 'rgb(150,150,150)'], [1, 'rgb(210,210,210)']]
        else:
            colorscale = [[0, 'gold'], [0.5, 'purple'], [1, 'mediumblue']]
        fig = go.Figure(data=[go.Parcats(dimensions=[*perf_dim, *feature_dim],
                                         line={'color': color, 'colorscale': colorscale}, bundlecolors=True,
                                         hoveron='category', hoverinfo='count+probability',
                                         arrangement='freeform')])
        fig.update_layout(margin=dict(l=20, r=20, b=10, t=40, pad=4))
        return fig


@app.callback(
    Output('df-feature-detail', "data"),
    Output('dict-bin-mapping2', "data"),
    Output('dict-bin-mapping3', "data"),
    Output('dict-clusters-metrics', "data"),
    Input('dict-patient-id', "data"),
    Input('json-selected-models', "data"),
    Input('dropdown-pcd-cluster', "value"),
    Input('dropdown-pcd-type', "value"),
    Input('dropdown-feature', "value"),
    Input('slice-slider', "value"),
    Input('df-metric-detail', "data")
)
def update_data_parcats_detail(json_patient_id, json_selected_models, pcd_type, number_clusters, selected_features,
                               current_slice, json_metric):
    if json_patient_id is None:
        raise PreventUpdate
    else:
        patient_id = json.loads(json_patient_id)
        number_clusters = int(number_clusters)
        # performance features
        selected_models = json.loads(json_selected_models)
        if selected_models is None or len(selected_models) == 0:
            return pd.DataFrame().to_json(), json.dumps({})
        if pcd_type == "model_pcd":
            df = pd.read_json(f"{DATA_PATH}/vs_gk_{patient_id}/radiomics_pred_2d.json")
            if selected_features == "firstorder":
                feat_list = ["Energy", "Skewness", "Kurtosis", "Variance", "Range", "MeanAbsoluteDeviation"]
            elif selected_features == "shape":
                feat_list = ["MeshSurface", "Sphericity", "Elongation", "MaximumDiameter"]
                selected_features = "shape2D"
            else:
                return pd.DataFrame(index=[0], columns=["use other feature"]).to_json(), None, None, None

            X = {}
            model_names = []
            for m in df.keys():
                if m in selected_models:
                    x = []
                    if current_slice in df[m].keys():
                        if type(df[m][current_slice]) == float and np.isnan(df[m][current_slice]):
                            continue
                        [x.append(float(df[m][current_slice][selected_features][f])) for f in feat_list]
                        model_names.append(m)
                        X[m] = x
            try:
                clustering = AgglomerativeClustering(n_clusters=number_clusters)
                y = clustering.fit_predict(list(X.values()))
                clusters = dict(zip(model_names, y))
            except ValueError as e:
                return pd.DataFrame(index=[0], columns=["cluster number invalid"]).to_json(), None, None, None

            df_merged = pd.DataFrame()
            for m in model_names:
                m_dict = {"model": m,
                          "cluster": int(clusters[m])}
                if current_slice in df[m].keys():
                    m_dict.update({f: df[m][current_slice][selected_features][f] for f in feat_list})
                else:
                    continue
                df_merged = df_merged.append(m_dict, ignore_index=True)
            df_merged = df_merged[["model", "cluster"] + feat_list]

            for fc in feat_list:
                vals = [float(v) for v in df_merged[fc].values.tolist()]
                if fc == "Skewness":
                    df_merged[f"{fc}"] = [1 if a <= 0 else 2 for a in vals]
                elif fc == "Kurtosis":
                    df_merged[f"{fc}"] = [1 if a <= 3 else 2 for a in vals]
                elif fc == "Elongation":
                    df_merged[f"{fc}"] = [1 if a <= np.mean(vals) else 2 for a in vals]
                elif fc == "Flatness":
                    df_merged[f"{fc}"] = [1 if a <= 0.5 else 2 for a in vals]
                elif fc == "Sphericity":
                    df_merged[f"{fc}"] = [1 if a <= np.mean(vals) else 2 for a in vals]
                else:
                    df_merged[f"{fc}"] = np.digitize(vals, bins=np.linspace(np.min(vals),
                                                                            np.nextafter(np.max(vals), np.inf),
                                                                            4))
            dict_performance_bins = {}
            dict_size_bins = {}

            # create averaged metric per cluster for order of clusters
            df_metric = pd.read_json(json_metric)
            clusters_metric = dict((k, []) for k in range(number_clusters))
            t = df_metric[df_metric["slice"] == int(current_slice)].iloc[0]
            for k, v in t.iteritems():
                if k in clusters.keys():
                    clusters_metric[clusters[k]].append(v)
            clusters_metric_avg = dict(zip(clusters_metric.keys(), [np.average(v) for v in clusters_metric.values()]))
        else:
            df_total = pd.read_json(
                f"{DATA_PATH}/vs_gk_{patient_id}/evaluation.json")
            df_performance = pd.DataFrame()
            values = [3, 2, 1]
            values_ascending = dict(enumerate(values, 1))
            values_descending = dict(enumerate(reversed(values), 1))
            dict_performance_bins = {}
            df_performance["id"] = [str(idx) for idx in range(len(df_total))]
            for d in ["only_tumor", "all"]:
                for m in ["dice", "assd"]:
                    df_tmp = pd.DataFrame()
                    for model in selected_models:
                        df_tmp[m + '_' + model] = df_total[f"VS_segm_{m}-{model}"]
                    df_performance[m + '_' + d] = df_tmp.mean(axis=1).values
                    bins = np.linspace(np.min(df_performance[m + '_' + d]),
                                       np.nextafter(np.max(df_performance[m + '_' + d]), np.inf), 4)
                    if "assd" in m:
                        df_performance[m + '_' + d] = np.vectorize(values_descending.get)(
                            np.digitize(df_performance[m + '_' + d], bins=bins))
                    else:
                        df_performance[m + '_' + d] = np.vectorize(values_ascending.get)(
                            np.digitize(df_performance[m + '_' + d], bins=bins))
                    if m + "_" + d not in dict_performance_bins.keys():
                        dict_performance_bins[m + "_" + d] = bins.tolist()
            # radiomics features
            with open(
                    f"{DATA_PATH}/vs_gk_{patient_id}/radiomics_gt_2d.json") as json_file:
                df_rad = json.load(json_file)
            df_radiomics = {"id": list(df_rad.keys())}
            feature_classes = list(df_rad[list(df_rad.keys())[0]].keys())
            for cl in feature_classes:
                cl_dict = {}
                for key in df_rad.keys():
                    cl_dict[key] = df_rad[str(key)][cl]
                tmp = {}
                for idx, d in cl_dict.items():
                    for f, vals in d.items():
                        if f in tmp.keys():
                            tmp[f] = tmp[f] + [vals]
                        else:
                            tmp[f] = [vals]
                df_radiomics[cl] = tmp
            df_sign = pd.DataFrame(columns=["id"])
            df_sign["id"] = df_radiomics["id"]
            for fc in feature_classes:
                for key, vals in df_radiomics[fc].items():
                    vals = [float(v) for v in vals]
                    if key == "Skewness":
                        df_sign[f"{fc}-{key}"] = [1 if a <= 0 else 2 for a in vals]
                    elif key == "Kurtosis":
                        df_sign[f"{fc}-{key}"] = [1 if a <= 3 else 2 for a in vals]
                    elif key == "Elongation":
                        df_sign[f"{fc}-{key}"] = [1 if a <= np.mean(vals) else 2 for a in vals]
                    elif key == "Flatness":
                        df_sign[f"{fc}-{key}"] = [1 if a <= 0.5 else 2 for a in vals]
                    elif key == "Sphericity":
                        df_sign[f"{fc}-{key}"] = [1 if a <= np.mean(vals) else 2 for a in vals]
                    else:
                        df_sign[f"{fc}-{key}"] = np.digitize(vals, bins=np.linspace(np.min(vals),
                                                                                    np.nextafter(np.max(vals), np.inf),
                                                                                    4))
            # volume features
            df_volume = pd.DataFrame()
            df = pd.read_json(f"{DATA_PATH}/vs_gk_{patient_id}/evaluation.json")
            for idx, row in df.iterrows():
                df_volume = df_volume.append({"id": str(row["slice"]),
                                              "tumor_presence": row["VS_class_gt"],
                                              "tumor_size_px": np.count_nonzero(cv2.drawContours(np.zeros((256, 256)),
                                                                                                 [np.array(s).astype(
                                                                                                     np.int64) for s in
                                                                                                     np.array(
                                                                                                         row[
                                                                                                             "VS_segm_gt"],
                                                                                                         dtype="object")],
                                                                                                 -1,
                                                                                                 (1),
                                                                                                 -1))},
                                             ignore_index=True)
            bins = np.linspace(np.min(df_volume[df_volume["tumor_size_px"] >= 1]["tumor_size_px"]),
                               np.nextafter(np.max(df_volume[df_volume["tumor_presence"] >= 1]["tumor_size_px"]),
                                            np.inf),
                               4)
            res = [0] * len(df_volume)
            res[np.where(df_volume["tumor_presence"] == 1)[0][0]:np.where(df_volume["tumor_presence"] == 1)[0][
                                                                     -1] + 1] = np.digitize(
                df_volume[df_volume["tumor_presence"] == 1]["tumor_size_px"], bins=bins)
            dict_size_bins = {"tumor_size": bins.tolist()}
            df_volume["tumor_size"] = np.array(res)
            df_volume["tumor_presence"] = df_volume["tumor_presence"].apply(lambda x: x + 1)
            # merge features
            df_merged = df_volume.merge(df_performance.merge(df_sign, on="id", how="left"), on="id", how="left")
            df_merged = df_merged.fillna(value=0)
            clusters_metric_avg = {}
        return df_merged.to_json(), json.dumps(dict_performance_bins), json.dumps(dict_size_bins), json.dumps(
            clusters_metric_avg)


@app.callback(
    Output('parcats-detail', "figure"),
    Input('df-feature-detail', "data"),
    Input('dict-bin-mapping2', "data"),
    Input('dict-bin-mapping3', "data"),
    Input('dropdown-error-metric', "value"),
    Input('radioitem-dataset', "value"),
    Input('dropdown-feature', "value"),
    Input('theme', "data"),
    Input('dropdown-pcd-cluster', "value"),
    Input('dropdown-pcd-type', "value"),
    Input('dict-clusters-metrics', "data"),
)
def update_parcats_detail(df_plot_json, dict_bin_mapping, dict_bin_mapping2, selected_metric, selected_dataset,
                          selected_features, theme, pcd_type, number_cluster, df_cluster_metric_json):
    if df_plot_json is None or selected_metric is None or selected_dataset is None:
        raise PreventUpdate
    else:
        lookup = {"DSC": "dice",
                  "ASSD": "assd",
                  "ACC": "dice",
                  "TNR": "dice",
                  "TPR": "dice"}
        selected_metric = lookup[selected_metric]
        if pcd_type == "model_pcd":
            fig = create_pcd_model(df_plot_json, number_cluster, theme, df_cluster_metric_json, selected_metric)
        else:
            fig = create_pcd_slice(df_plot_json, dict_bin_mapping, dict_bin_mapping2, selected_metric, selected_dataset,
                                   selected_features, theme)
        return fig


def create_pcd_model(df_plot_json, number_cluster, theme, df_cluster_metric_json, metric):
    df_plot = pd.read_json(df_plot_json)
    number_cluster = int(number_cluster)

    if len(df_plot) == 0:
        return fig_no_model_selected
    if len(df_plot) == 1 and "use other feature" in df_plot.columns:
        return fig_use_radiomics_features
    if len(df_plot) == 1 and "cluster number invalid" in df_plot.columns:
        return fig_cluster_number

    # create averaged metric per cluster for order of clusters
    dict_cluster_metric = json.loads(df_cluster_metric_json)
    if metric == "dice":
        to_sort = [-x for x in dict_cluster_metric.values()]
    else:
        to_sort = list(dict_cluster_metric.values())
    sorted_idx = np.argsort(to_sort)
    k = list(dict_cluster_metric.keys())
    v = list(dict_cluster_metric.values())
    idx_mapping = dict(zip(k, sorted_idx))
    cluster_value_mapping = {k[i]: v[i] for i in sorted_idx}
    df_plot["cluster"] = df_plot["cluster"].apply(lambda x: idx_mapping[str(x)])

    # Create dimensions
    lookup_dict = get_lookup_dict_detail_2()
    k = list(dict_cluster_metric.keys())

    cluster_lookup = dict(ticktext=[f"{i_} - {round(v[i], 3)}" for i_, i in enumerate(sorted_idx)],
                          categoryarray=list(range(number_cluster)),
                          label="Cluster")
    idx_mapping = dict(zip(k, sorted_idx))
    cluster_dim = go.parcats.Dimension(values=df_plot["cluster"], **cluster_lookup)
    feature_dim = []
    for f in list(df_plot.keys())[2:]:
        lookup_dict_ = lookup_dict[f]
        diff = list(set(lookup_dict[f]["categoryarray"]) - set(np.unique(df_plot[f].values)))
        for d in diff:
            idx = np.where(np.array(lookup_dict[f]["categoryarray"]) == d)[0][0]
            lookup_dict[f]["ticktext"].pop(idx)
            lookup_dict[f]["categoryarray"].pop(idx)
        if "label" in lookup_dict[f].keys():
            feature_dim.append(go.parcats.Dimension(values=df_plot[f], **lookup_dict_))
        else:
            feature_dim.append(go.parcats.Dimension(values=df_plot[f], label=f, **lookup_dict_))
    # Create parcats trace
    color = df_plot["cluster"]  # cluster_values
    if json.loads(theme) == "gray":
        colorscale = [[0, 'rgb(42,42,42)'], [0.5, 'rgb(150,150,150)'], [1, 'rgb(210,210,210)']]
    else:
        colorscale = [[0, 'gold'], [0.5, 'purple'], [1, 'mediumblue']]
    fig = go.Figure(data=[go.Parcats(dimensions=[cluster_dim, *feature_dim],
                                     line={'color': color, 'colorscale': colorscale}, bundlecolors=True,
                                     hoveron='category', hoverinfo='count+probability',
                                     arrangement='freeform')])
    fig.update_layout(margin=dict(l=5,
                                  r=5,
                                  b=5,
                                  t=20,
                                  pad=4)
                      )
    return fig


def create_pcd_slice(df_plot_json, dict_bin_mapping, dict_bin_mapping2, metric, selected_dataset,
                     selected_features, theme):
    df_new = pd.read_json(df_plot_json)
    bin_mapping = json.loads(dict_bin_mapping)
    bin_mapping_size = json.loads(dict_bin_mapping2)
    if len(df_new) == 0:
        return fig_no_model_selected
    # define plot df
    feature_list = get_feature_lookup_table(selected_features, metric, selected_dataset, type_level="detail")
    df_plot = pd.DataFrame()
    df_plot["id"] = df_new["id"]
    df_plot["tumor_presence"] = df_new["tumor_presence"]
    df_plot["tumor_size"] = df_new["tumor_size"]
    performance_col = metric + "_" + selected_dataset
    df_plot[performance_col] = df_new[performance_col].values
    for feature in feature_list:
        df_plot[feature] = df_new[[c for c in df_new.columns if feature in c][0]].values
    plot_cols = df_plot.columns.to_list()
    plot_cols.remove("id")
    # Create dimensions
    performance_col = metric + "_" + selected_dataset
    lookup_dict = get_lookup_dict_detail(bin_mapping, bin_mapping_size)
    # Create dimensions
    perf_dim = [go.parcats.Dimension(values=df_plot[performance_col], **lookup_dict[performance_col]),
                go.parcats.Dimension(values=df_plot["tumor_presence"], **lookup_dict["tumor_presence"]),
                go.parcats.Dimension(values=df_plot["tumor_size"], **lookup_dict["tumor_size"]),
                ]
    for dim in perf_dim:
        if len(dim["ticktext"]) != len(set(dim["values"])):
            return fig_not_possible
    plot_cols.remove("tumor_presence")
    plot_cols.remove("tumor_size")
    plot_cols.remove(performance_col)
    feature_dim = []
    for f in plot_cols:
        if "label" in lookup_dict[f].keys():
            feature_dim.append(go.parcats.Dimension(values=df_plot[f], **lookup_dict[f]))
        else:
            feature_dim.append(go.parcats.Dimension(values=df_plot[f], label=f, **lookup_dict[f]))
    for dim in feature_dim:
        if len(dim["ticktext"]) != len(set(dim["values"])):
            return fig_not_possible_other_features
    # Create parcats trace
    color = df_plot[performance_col]
    if json.loads(theme) == "gray":
        colorscale = [[0, 'rgb(42,42,42)'], [0.5, 'rgb(150,150,150)'], [1, 'rgb(210,210,210)']]
    else:
        colorscale = [[0, 'gold'], [0.5, 'purple'], [1, 'mediumblue']]
    fig = go.Figure(data=[go.Parcats(dimensions=[*perf_dim, *feature_dim],
                                     line={'color': color, 'colorscale': colorscale}, bundlecolors=True,
                                     hoveron='category', hoverinfo='count+probability',
                                     arrangement='freeform')])
    fig.update_layout(margin=dict(l=20, r=30, b=10, t=40, pad=4))
    return fig


if __name__ == "__main__":
    app.run_server(host='0.0.0.0', port=8060, debug=True)
