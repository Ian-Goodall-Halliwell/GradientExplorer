from mmap import ALLOCATIONGRANULARITY
from dash.dash import no_update
from flask import Flask
from os.path import exists, join, dirname, abspath
from os import makedirs, mkdir
from nilearn.input_data import NiftiMasker
from dash_bootstrap_components.themes import COSMO
from flask import send_from_directory
from dash import Dash
import dash_bootstrap_components as dbc
from dash import html as html
from dash import dcc as dcc
from dash.dependencies import Input, Output, State
from base64 import decodebytes
from zipfile import ZipFile
from os import remove, walk, listdir, sep
from nilearn.masking import apply_mask, compute_background_mask
from nilearn.image import resample_to_img
from nibabel import load as niload
from scipy.stats import pearsonr
from numpy import squeeze, array, abs
from wordcloud import WordCloud
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib.cm import RdBu_r
from matplotlib.colors import to_hex
from shutil import rmtree
from plotly.io import read_json
from plotly.graph_objects import Scatter, Scatter3d, Figure
from plotly.colors import qualitative
import plotly_express as px
import pandas as pd
from sklearn import preprocessing
import csv
from nilearn.plotting import plot_img_comparison

red_button_style = {"color": "primary"}

UPLOAD_DIRECTORY = "Downloads"
if not exists(UPLOAD_DIRECTORY):
    makedirs(UPLOAD_DIRECTORY)

server = Flask(__name__)
app = Dash(
    server=server,
    external_stylesheets=[COSMO],
    suppress_callback_exceptions=True,
    include_assets_files=False,
    title="Gradient Explorer",
    update_title="Processing...",
)
app._favicon = "favicon.ico"
application = app.server


@server.before_first_request
def before_first_request():
    if exists("assets") == False:
        mkdir("assets")
    dir_name = "assets"
    test = listdir(dir_name)
    for item in test:
        if item.endswith(".png"):
            remove(join(dir_name, item))


@server.route("/download/<path:path>")
def download(path):
    """Serve a file from the upload directory."""
    return send_from_directory(UPLOAD_DIRECTORY, path, as_attachment=True)


app.layout = dbc.Container(
    [
        dbc.Spinner(
            [
                html.Div([""], id="loading-output4"),
                html.Div([""], id="loading-output3"),
            ],
            fullscreen=True,
        ),
        html.H1("Gradient Explorer"),
        html.H5(
            dcc.Markdown(
                """
            Upload an fMRI file or .zip archive containing fMRI files in a nibabel-compatible format and wait for the file(s) to be processed. When the processing is done, click the blue button below. Uploading single files will also generate wordclouds of the top 10 closest terms/topics. As of right now, these are stored in your browser's cache, and require you to **clear your cache to generate new wordclouds**. If you have any problems that can't be fixed by a refresh please email me at <goodallhalliwell.i@queensu.ca>.
        """
            )
        ),
        dcc.Upload(
            id="upload-data",
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
            multiple=True,
            children=dbc.Spinner(
                html.Div(
                    ["Click or drag a file here to upload it."], id="loading-output",
                )
            ),
        ),
        dcc.Store(id="session", storage_type="session"),
        dcc.Store(id="session2", storage_type="local"),
        dcc.Store(id="session3", storage_type="local"),
        dcc.Store(id="session4", storage_type="local"),
        dcc.Store(id="geng", storage_type="session"),
        dcc.Store(id="store1", storage_type="session"),
        dcc.Store(id="cl", storage_type="session"),
        dcc.Store(id="iszipped", storage_type="session"),
        dcc.Store(id="showbar1", storage_type="session"),
        dcc.Store(id="showbar2", storage_type="session"),
        dcc.Store(id="showbar3", storage_type="session"),
        dcc.Store(id="showbar4", storage_type="session"),
        dcc.Store(id="spare", storage_type="session"),
        html.Hr(),
        dbc.Button(
            color="primary",
            block=True,
            id="button",
            className="mb-3",
            disabled=True,
            children=dbc.Spinner(
                html.Div(["Place scan into gradient space"], id="loading-output2",)
            ),
        ),
        dbc.Tabs(
            [
                dbc.Tab(label="50-Topic Dataset", tab_id="50-Topic Dataset"),
                dbc.Tab(label="100-Topic Dataset", tab_id="100-Topic Dataset"),
                dbc.Tab(label="200-Topic Dataset", tab_id="200-Topic Dataset"),
                dbc.Tab(label="400-Topic Dataset", tab_id="400-Topic Dataset"),
                dbc.Tab(label="Full Term Dataset", tab_id="Full Term Dataset"),
            ],
            id="tabs",
            active_tab="50-Topic Dataset",
        ),
        html.Div(id="tab-content", className="p-4",),
    ],
    style={"max-width": "95vw", "width": "80vw"},
)


@app.callback(
    Output("loading-output4", "children"),
    Output("tab-content", "children"),
    Input("tabs", "active_tab"),
    [Input("store1", "data"),],
    Input("geng", "data"),
    Input("cl", "data"),
    Input("showbar1", "data"),
    Input("showbar2", "data"),
    Input("showbar3", "data"),
    Input("showbar4", "data"),
)
def render_tab_content(
    active_tab, data1, displayoptions, cldata, g1state, g2state, g3state, g4state
):
    if active_tab and data1 is not None:
        if active_tab == "50-Topic Dataset":
            print(checkif2(displayoptions=g1state))
            return (
                f"",
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Col(
                                    [
                                        html.Div(
                                            [
                                                html.Img(
                                                    id="element-to-hide",
                                                    src=cldata,
                                                    style={"textAlign": "center"},
                                                )
                                            ],
                                            style={
                                                "display": checkif(displayoptions),
                                                "textAlign": "center",
                                            },
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(html.Div([data1[0]])),
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            dcc.Graph(
                                                                figure=(
                                                                    px.bar(
                                                                        title="Click on a point",
                                                                        orientation="h",
                                                                        color_continuous_scale="plasma_r",
                                                                        height=900,
                                                                    )
                                                                ).update_layout(
                                                                    title_x=0.5,
                                                                    autosize=True,
                                                                    xaxis=dict(
                                                                        showgrid=False,  # thin lines in the background
                                                                        zeroline=False,  # thick line at x=0
                                                                        visible=False
                                                                        # numbers below
                                                                    ),
                                                                    yaxis=dict(
                                                                        showgrid=False,  # thin lines in the background
                                                                        zeroline=False,  # thick line at x=0
                                                                        visible=False,  # numbers below
                                                                    ),
                                                                )
                                                            ),
                                                        ],
                                                        id="1graph1",
                                                        style={
                                                            "display": checkif2(
                                                                displayoptions=g1state
                                                            )
                                                        },
                                                    ),
                                                    width=3,
                                                ),
                                            ]
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(html.Div([data1[1]])),
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            dcc.Graph(
                                                                figure=(
                                                                    px.bar(
                                                                        title="Click on a point",
                                                                        orientation="h",
                                                                        color_continuous_scale="plasma_r",
                                                                        height=900,
                                                                    )
                                                                ).update_layout(
                                                                    title_x=0.5,
                                                                    autosize=True,
                                                                    xaxis=dict(
                                                                        showgrid=False,  # thin lines in the background
                                                                        zeroline=False,  # thick line at x=0
                                                                        visible=False
                                                                        # numbers below
                                                                    ),
                                                                    yaxis=dict(
                                                                        showgrid=False,  # thin lines in the background
                                                                        zeroline=False,  # thick line at x=0
                                                                        visible=False,  # numbers below
                                                                    ),
                                                                )
                                                            )
                                                        ],
                                                        id="1graph2",
                                                        style={
                                                            "display": checkif2(
                                                                displayoptions=g1state
                                                            )
                                                        },
                                                    ),
                                                    width=3,
                                                ),
                                            ]
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(html.Div([data1[2]])),
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            dcc.Graph(
                                                                figure=(
                                                                    px.bar(
                                                                        title="Click on a point",
                                                                        orientation="h",
                                                                        color_continuous_scale="plasma_r",
                                                                        height=900,
                                                                    )
                                                                ).update_layout(
                                                                    title_x=0.5,
                                                                    autosize=True,
                                                                    xaxis=dict(
                                                                        showgrid=False,  # thin lines in the background
                                                                        zeroline=False,  # thick line at x=0
                                                                        visible=False
                                                                        # numbers below
                                                                    ),
                                                                    yaxis=dict(
                                                                        showgrid=False,  # thin lines in the background
                                                                        zeroline=False,  # thick line at x=0
                                                                        visible=False,  # numbers below
                                                                    ),
                                                                )
                                                            )
                                                        ],
                                                        id="1graph3",
                                                        style={
                                                            "display": checkif2(
                                                                displayoptions=g1state
                                                            )
                                                        },
                                                    ),
                                                    width=3,
                                                ),
                                            ]
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(html.Div([data1[3]])),
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            dcc.Graph(
                                                                figure=(
                                                                    px.bar(
                                                                        title="Click on a point",
                                                                        orientation="h",
                                                                        color_continuous_scale="plasma_r",
                                                                        height=900,
                                                                    )
                                                                ).update_layout(
                                                                    title_x=0.5,
                                                                    autosize=True,
                                                                    xaxis=dict(
                                                                        showgrid=False,  # thin lines in the background
                                                                        zeroline=False,  # thick line at x=0
                                                                        visible=False
                                                                        # numbers below
                                                                    ),
                                                                    yaxis=dict(
                                                                        showgrid=False,  # thin lines in the background
                                                                        zeroline=False,  # thick line at x=0
                                                                        visible=False,  # numbers below
                                                                    ),
                                                                )
                                                            )
                                                        ],
                                                        id="1graph4",
                                                        style={
                                                            "display": checkif2(
                                                                displayoptions=g1state
                                                            )
                                                        },
                                                    ),
                                                    width=3,
                                                ),
                                            ]
                                        ),
                                    ]
                                )
                            ],
                            align="center",  # width={"size": 10}
                        ),
                    ],
                ),
            )

        elif active_tab == "100-Topic Dataset":
            return (
                f"",
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Col(
                                    [
                                        html.Div(
                                            [
                                                html.Img(
                                                    id="element-to-hide",
                                                    src=cldata,
                                                    style={"textAlign": "center"},
                                                )
                                            ],
                                            style={
                                                "display": checkif(displayoptions),
                                                "textAlign": "center",
                                            },
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(html.Div([data1[0]])),
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            dcc.Graph(
                                                                figure=(
                                                                    px.bar(
                                                                        title="Click on a point",
                                                                        orientation="h",
                                                                        color_continuous_scale="plasma_r",
                                                                        height=900,
                                                                    )
                                                                ).update_layout(
                                                                    title_x=0.5,
                                                                    autosize=True,
                                                                    xaxis=dict(
                                                                        showgrid=False,  # thin lines in the background
                                                                        zeroline=False,  # thick line at x=0
                                                                        visible=False
                                                                        # numbers below
                                                                    ),
                                                                    yaxis=dict(
                                                                        showgrid=False,  # thin lines in the background
                                                                        zeroline=False,  # thick line at x=0
                                                                        visible=False,  # numbers below
                                                                    ),
                                                                )
                                                            ),
                                                        ],
                                                        id="1graph1",
                                                        style={
                                                            "display": checkif2(
                                                                displayoptions=g1state
                                                            )
                                                        },
                                                    ),
                                                    width=3,
                                                ),
                                            ]
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(html.Div([data1[1]])),
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            dcc.Graph(
                                                                figure=(
                                                                    px.bar(
                                                                        title="Click on a point",
                                                                        orientation="h",
                                                                        color_continuous_scale="plasma_r",
                                                                        height=900,
                                                                    )
                                                                ).update_layout(
                                                                    title_x=0.5,
                                                                    autosize=True,
                                                                    xaxis=dict(
                                                                        showgrid=False,  # thin lines in the background
                                                                        zeroline=False,  # thick line at x=0
                                                                        visible=False
                                                                        # numbers below
                                                                    ),
                                                                    yaxis=dict(
                                                                        showgrid=False,  # thin lines in the background
                                                                        zeroline=False,  # thick line at x=0
                                                                        visible=False,  # numbers below
                                                                    ),
                                                                )
                                                            )
                                                        ],
                                                        id="1graph2",
                                                        style={
                                                            "display": checkif2(
                                                                displayoptions=g1state
                                                            )
                                                        },
                                                    ),
                                                    width=3,
                                                ),
                                            ]
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(html.Div([data1[2]])),
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            dcc.Graph(
                                                                figure=(
                                                                    px.bar(
                                                                        title="Click on a point",
                                                                        orientation="h",
                                                                        color_continuous_scale="plasma_r",
                                                                        height=900,
                                                                    )
                                                                ).update_layout(
                                                                    title_x=0.5,
                                                                    autosize=True,
                                                                    xaxis=dict(
                                                                        showgrid=False,  # thin lines in the background
                                                                        zeroline=False,  # thick line at x=0
                                                                        visible=False
                                                                        # numbers below
                                                                    ),
                                                                    yaxis=dict(
                                                                        showgrid=False,  # thin lines in the background
                                                                        zeroline=False,  # thick line at x=0
                                                                        visible=False,  # numbers below
                                                                    ),
                                                                )
                                                            )
                                                        ],
                                                        id="1graph3",
                                                        style={
                                                            "display": checkif2(
                                                                displayoptions=g1state
                                                            )
                                                        },
                                                    ),
                                                    width=3,
                                                ),
                                            ]
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(html.Div([data1[3]])),
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            dcc.Graph(
                                                                figure=(
                                                                    px.bar(
                                                                        title="Click on a point",
                                                                        orientation="h",
                                                                        color_continuous_scale="plasma_r",
                                                                        height=900,
                                                                    )
                                                                ).update_layout(
                                                                    title_x=0.5,
                                                                    autosize=True,
                                                                    xaxis=dict(
                                                                        showgrid=False,  # thin lines in the background
                                                                        zeroline=False,  # thick line at x=0
                                                                        visible=False
                                                                        # numbers below
                                                                    ),
                                                                    yaxis=dict(
                                                                        showgrid=False,  # thin lines in the background
                                                                        zeroline=False,  # thick line at x=0
                                                                        visible=False,  # numbers below
                                                                    ),
                                                                )
                                                            )
                                                        ],
                                                        id="1graph4",
                                                        style={
                                                            "display": checkif2(
                                                                displayoptions=g1state
                                                            )
                                                        },
                                                    ),
                                                    width=3,
                                                ),
                                            ]
                                        ),
                                    ]
                                )
                            ],
                            align="center",  # width={"size": 10}
                        ),
                    ],
                ),
            )

        elif active_tab == "200-Topic Dataset":
            return (
                f"",
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Col(
                                    [
                                        html.Div(
                                            [
                                                html.Img(
                                                    id="element-to-hide",
                                                    src=cldata,
                                                    style={"textAlign": "center"},
                                                )
                                            ],
                                            style={
                                                "display": checkif(displayoptions),
                                                "textAlign": "center",
                                            },
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(html.Div([data1[0]])),
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            dcc.Graph(
                                                                figure=(
                                                                    px.bar(
                                                                        title="Click on a point",
                                                                        orientation="h",
                                                                        color_continuous_scale="plasma_r",
                                                                        height=900,
                                                                    )
                                                                ).update_layout(
                                                                    title_x=0.5,
                                                                    autosize=True,
                                                                    xaxis=dict(
                                                                        showgrid=False,  # thin lines in the background
                                                                        zeroline=False,  # thick line at x=0
                                                                        visible=False
                                                                        # numbers below
                                                                    ),
                                                                    yaxis=dict(
                                                                        showgrid=False,  # thin lines in the background
                                                                        zeroline=False,  # thick line at x=0
                                                                        visible=False,  # numbers below
                                                                    ),
                                                                )
                                                            ),
                                                        ],
                                                        id="1graph1",
                                                        style={
                                                            "display": checkif2(
                                                                displayoptions=g1state
                                                            )
                                                        },
                                                    ),
                                                    width=3,
                                                ),
                                            ]
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(html.Div([data1[1]])),
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            dcc.Graph(
                                                                figure=(
                                                                    px.bar(
                                                                        title="Click on a point",
                                                                        orientation="h",
                                                                        color_continuous_scale="plasma_r",
                                                                        height=900,
                                                                    )
                                                                ).update_layout(
                                                                    title_x=0.5,
                                                                    autosize=True,
                                                                    xaxis=dict(
                                                                        showgrid=False,  # thin lines in the background
                                                                        zeroline=False,  # thick line at x=0
                                                                        visible=False
                                                                        # numbers below
                                                                    ),
                                                                    yaxis=dict(
                                                                        showgrid=False,  # thin lines in the background
                                                                        zeroline=False,  # thick line at x=0
                                                                        visible=False,  # numbers below
                                                                    ),
                                                                )
                                                            )
                                                        ],
                                                        id="1graph2",
                                                        style={
                                                            "display": checkif2(
                                                                displayoptions=g1state
                                                            )
                                                        },
                                                    ),
                                                    width=3,
                                                ),
                                            ]
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(html.Div([data1[2]])),
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            dcc.Graph(
                                                                figure=(
                                                                    px.bar(
                                                                        title="Click on a point",
                                                                        orientation="h",
                                                                        color_continuous_scale="plasma_r",
                                                                        height=900,
                                                                    )
                                                                ).update_layout(
                                                                    title_x=0.5,
                                                                    autosize=True,
                                                                    xaxis=dict(
                                                                        showgrid=False,  # thin lines in the background
                                                                        zeroline=False,  # thick line at x=0
                                                                        visible=False
                                                                        # numbers below
                                                                    ),
                                                                    yaxis=dict(
                                                                        showgrid=False,  # thin lines in the background
                                                                        zeroline=False,  # thick line at x=0
                                                                        visible=False,  # numbers below
                                                                    ),
                                                                )
                                                            )
                                                        ],
                                                        id="1graph3",
                                                        style={
                                                            "display": checkif2(
                                                                displayoptions=g1state
                                                            )
                                                        },
                                                    ),
                                                    width=3,
                                                ),
                                            ]
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(html.Div([data1[3]])),
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            dcc.Graph(
                                                                figure=(
                                                                    px.bar(
                                                                        title="Click on a point",
                                                                        orientation="h",
                                                                        color_continuous_scale="plasma_r",
                                                                        height=900,
                                                                    )
                                                                ).update_layout(
                                                                    title_x=0.5,
                                                                    autosize=True,
                                                                    xaxis=dict(
                                                                        showgrid=False,  # thin lines in the background
                                                                        zeroline=False,  # thick line at x=0
                                                                        visible=False
                                                                        # numbers below
                                                                    ),
                                                                    yaxis=dict(
                                                                        showgrid=False,  # thin lines in the background
                                                                        zeroline=False,  # thick line at x=0
                                                                        visible=False,  # numbers below
                                                                    ),
                                                                )
                                                            )
                                                        ],
                                                        id="1graph4",
                                                        style={
                                                            "display": checkif2(
                                                                displayoptions=g1state
                                                            )
                                                        },
                                                    ),
                                                    width=3,
                                                ),
                                            ]
                                        ),
                                    ]
                                )
                            ],
                            align="center",  # width={"size": 10}
                        ),
                    ],
                ),
            )

        elif active_tab == "400-Topic Dataset":
            return (
                f"",
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Col(
                                    [
                                        html.Div(
                                            [
                                                html.Img(
                                                    id="element-to-hide",
                                                    src=cldata,
                                                    style={"textAlign": "center"},
                                                )
                                            ],
                                            style={
                                                "display": checkif(displayoptions),
                                                "textAlign": "center",
                                            },
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(html.Div([data1[0]])),
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            dcc.Graph(
                                                                figure=(
                                                                    px.bar(
                                                                        title="Click on a point",
                                                                        orientation="h",
                                                                        color_continuous_scale="plasma_r",
                                                                        height=900,
                                                                    )
                                                                ).update_layout(
                                                                    title_x=0.5,
                                                                    autosize=True,
                                                                    xaxis=dict(
                                                                        showgrid=False,  # thin lines in the background
                                                                        zeroline=False,  # thick line at x=0
                                                                        visible=False
                                                                        # numbers below
                                                                    ),
                                                                    yaxis=dict(
                                                                        showgrid=False,  # thin lines in the background
                                                                        zeroline=False,  # thick line at x=0
                                                                        visible=False,  # numbers below
                                                                    ),
                                                                )
                                                            ),
                                                        ],
                                                        id="1graph1",
                                                        style={
                                                            "display": checkif2(
                                                                displayoptions=g1state
                                                            )
                                                        },
                                                    ),
                                                    width=3,
                                                ),
                                            ]
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(html.Div([data1[1]])),
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            dcc.Graph(
                                                                figure=(
                                                                    px.bar(
                                                                        title="Click on a point",
                                                                        orientation="h",
                                                                        color_continuous_scale="plasma_r",
                                                                        height=900,
                                                                    )
                                                                ).update_layout(
                                                                    title_x=0.5,
                                                                    autosize=True,
                                                                    xaxis=dict(
                                                                        showgrid=False,  # thin lines in the background
                                                                        zeroline=False,  # thick line at x=0
                                                                        visible=False
                                                                        # numbers below
                                                                    ),
                                                                    yaxis=dict(
                                                                        showgrid=False,  # thin lines in the background
                                                                        zeroline=False,  # thick line at x=0
                                                                        visible=False,  # numbers below
                                                                    ),
                                                                )
                                                            )
                                                        ],
                                                        id="1graph2",
                                                        style={
                                                            "display": checkif2(
                                                                displayoptions=g1state
                                                            )
                                                        },
                                                    ),
                                                    width=3,
                                                ),
                                            ]
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(html.Div([data1[2]])),
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            dcc.Graph(
                                                                figure=(
                                                                    px.bar(
                                                                        title="Click on a point",
                                                                        orientation="h",
                                                                        color_continuous_scale="plasma_r",
                                                                        height=900,
                                                                    )
                                                                ).update_layout(
                                                                    title_x=0.5,
                                                                    autosize=True,
                                                                    xaxis=dict(
                                                                        showgrid=False,  # thin lines in the background
                                                                        zeroline=False,  # thick line at x=0
                                                                        visible=False
                                                                        # numbers below
                                                                    ),
                                                                    yaxis=dict(
                                                                        showgrid=False,  # thin lines in the background
                                                                        zeroline=False,  # thick line at x=0
                                                                        visible=False,  # numbers below
                                                                    ),
                                                                )
                                                            )
                                                        ],
                                                        id="1graph3",
                                                        style={
                                                            "display": checkif2(
                                                                displayoptions=g1state
                                                            )
                                                        },
                                                    ),
                                                    width=3,
                                                ),
                                            ]
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(html.Div([data1[3]])),
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            dcc.Graph(
                                                                figure=(
                                                                    px.bar(
                                                                        title="Click on a point",
                                                                        orientation="h",
                                                                        color_continuous_scale="plasma_r",
                                                                        height=900,
                                                                    )
                                                                ).update_layout(
                                                                    title_x=0.5,
                                                                    autosize=True,
                                                                    xaxis=dict(
                                                                        showgrid=False,  # thin lines in the background
                                                                        zeroline=False,  # thick line at x=0
                                                                        visible=False
                                                                        # numbers below
                                                                    ),
                                                                    yaxis=dict(
                                                                        showgrid=False,  # thin lines in the background
                                                                        zeroline=False,  # thick line at x=0
                                                                        visible=False,  # numbers below
                                                                    ),
                                                                )
                                                            )
                                                        ],
                                                        id="1graph4",
                                                        style={
                                                            "display": checkif2(
                                                                displayoptions=g1state
                                                            )
                                                        },
                                                    ),
                                                    width=3,
                                                ),
                                            ]
                                        ),
                                    ]
                                )
                            ],
                            align="center",  # width={"size": 10}
                        ),
                    ],
                ),
            )

        elif active_tab == "Full Term Dataset":
            return (
                f"",
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Col(
                                    [
                                        html.Div(
                                            [
                                                html.Img(
                                                    id="element-to-hide",
                                                    src=cldata,
                                                    style={"textAlign": "center"},
                                                )
                                            ],
                                            style={
                                                "display": checkif(displayoptions),
                                                "textAlign": "center",
                                            },
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(html.Div([data1[0]])),
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            dcc.Graph(
                                                                figure=(
                                                                    px.bar(
                                                                        title="Click on a point",
                                                                        orientation="h",
                                                                        color_continuous_scale="plasma_r",
                                                                        height=900,
                                                                    )
                                                                ).update_layout(
                                                                    title_x=0.5,
                                                                    autosize=True,
                                                                    xaxis=dict(
                                                                        showgrid=False,  # thin lines in the background
                                                                        zeroline=False,  # thick line at x=0
                                                                        visible=False
                                                                        # numbers below
                                                                    ),
                                                                    yaxis=dict(
                                                                        showgrid=False,  # thin lines in the background
                                                                        zeroline=False,  # thick line at x=0
                                                                        visible=False,  # numbers below
                                                                    ),
                                                                )
                                                            ),
                                                        ],
                                                        id="1graph1",
                                                        style={
                                                            "display": checkif2(
                                                                displayoptions=g1state
                                                            )
                                                        },
                                                    ),
                                                    width=3,
                                                ),
                                            ]
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(html.Div([data1[1]])),
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            dcc.Graph(
                                                                figure=(
                                                                    px.bar(
                                                                        title="Click on a point",
                                                                        orientation="h",
                                                                        color_continuous_scale="plasma_r",
                                                                        height=900,
                                                                    )
                                                                ).update_layout(
                                                                    title_x=0.5,
                                                                    autosize=True,
                                                                    xaxis=dict(
                                                                        showgrid=False,  # thin lines in the background
                                                                        zeroline=False,  # thick line at x=0
                                                                        visible=False
                                                                        # numbers below
                                                                    ),
                                                                    yaxis=dict(
                                                                        showgrid=False,  # thin lines in the background
                                                                        zeroline=False,  # thick line at x=0
                                                                        visible=False,  # numbers below
                                                                    ),
                                                                )
                                                            )
                                                        ],
                                                        id="1graph2",
                                                        style={
                                                            "display": checkif2(
                                                                displayoptions=g1state
                                                            )
                                                        },
                                                    ),
                                                    width=3,
                                                ),
                                            ]
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(html.Div([data1[2]])),
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            dcc.Graph(
                                                                figure=(
                                                                    px.bar(
                                                                        title="Click on a point",
                                                                        orientation="h",
                                                                        color_continuous_scale="plasma_r",
                                                                        height=900,
                                                                    )
                                                                ).update_layout(
                                                                    title_x=0.5,
                                                                    autosize=True,
                                                                    xaxis=dict(
                                                                        showgrid=False,  # thin lines in the background
                                                                        zeroline=False,  # thick line at x=0
                                                                        visible=False
                                                                        # numbers below
                                                                    ),
                                                                    yaxis=dict(
                                                                        showgrid=False,  # thin lines in the background
                                                                        zeroline=False,  # thick line at x=0
                                                                        visible=False,  # numbers below
                                                                    ),
                                                                )
                                                            )
                                                        ],
                                                        id="1graph3",
                                                        style={
                                                            "display": checkif2(
                                                                displayoptions=g1state
                                                            )
                                                        },
                                                    ),
                                                    width=3,
                                                ),
                                            ]
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(html.Div([data1[3]])),
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            dcc.Graph(
                                                                figure=(
                                                                    px.bar(
                                                                        title="Click on a point",
                                                                        orientation="h",
                                                                        color_continuous_scale="plasma_r",
                                                                        height=900,
                                                                    )
                                                                ).update_layout(
                                                                    title_x=0.5,
                                                                    autosize=True,
                                                                    xaxis=dict(
                                                                        showgrid=False,  # thin lines in the background
                                                                        zeroline=False,  # thick line at x=0
                                                                        visible=False
                                                                        # numbers below
                                                                    ),
                                                                    yaxis=dict(
                                                                        showgrid=False,  # thin lines in the background
                                                                        zeroline=False,  # thick line at x=0
                                                                        visible=False,  # numbers below
                                                                    ),
                                                                )
                                                            )
                                                        ],
                                                        id="1graph4",
                                                        style={
                                                            "display": checkif2(
                                                                displayoptions=g1state
                                                            )
                                                        },
                                                    ),
                                                    width=3,
                                                ),
                                            ]
                                        ),
                                    ]
                                )
                            ],
                            align="center",  # width={"size": 10}
                        ),
                    ],
                ),
            )

    return f"", "No tab selected"


def checkif(displayoptions):
    if displayoptions == 0:
        return "none"
    elif displayoptions == 1:
        return "block"


def checkif2(displayoptions=None):
    if displayoptions == 0 or None:
        visible = "none"
        return visible
    elif displayoptions == 1:
        visible = "block"
        return visible


@app.callback(
    [
        Output("1graph1", "children"),
        Output("1graph2", "children"),
        Output("1graph3", "children"),
        Output("1graph4", "children"),
    ],
    Output("showbar1", "data"),
    Output("showbar2", "data"),
    Output("showbar3", "data"),
    Output("showbar4", "data"),
    Output("graph1col1", "figure"),
    Output("graph2col1", "figure"),
    Output("graph3col1", "figure"),
    Output("graph4col1", "figure"),
    # Output("store1", "data"),
    # Output("store1", "data"),
    Input("graph1col1", "clickData"),
    Input("graph2col1", "clickData"),
    Input("graph3col1", "clickData"),
    Input("graph4col1", "clickData"),
    Input("tabs", "active_tab"),
    Input("graph1col1", "figure"),
    Input("graph2col1", "figure"),
    Input("graph3col1", "figure"),
    Input("graph4col1", "figure"),
)
def graphupdate(g1c1, g2c1, g3c1, g4c1, tab, graph1, graph2, graph3, graph4):
    gdict = {1: g1c1, 2: g2c1, 3: g3c1, 4: g4c1}
    figlist = {"0": None, "1": None, "2": None, "3": None}
    flis = []
    colfigdict = {"0": graph1, "1": graph2, "2": graph3, "3": graph4}
    print(gdict)
    figdict1 = {
        "Full Term Dataset": [
            "v5-fulldataset.json",
            "1,2v5-fulldataset.json",
            "1,3v5-fulldataset.json",
            "2,3v5-fulldataset.json",
        ],
        "100-Topic Dataset": [
            "v5-topics-100.json",
            "1,2v5-topics-100.json",
            "1,3v5-topics-100.json",
            "2,3v5-topics-100.json",
        ],
        "200-Topic Dataset": [
            "v5-topics-200.json",
            "1,2v5-topics-200.json",
            "1,3v5-topics-200.json",
            "2,3v5-topics-200.json",
        ],
        "400-Topic Dataset": [
            "v5-topics-400.json",
            "1,2v5-topics-400.json",
            "1,3v5-topics-400.json",
            "2,3v5-topics-400.json",
        ],
        "50-Topic Dataset": [
            "v5-topics-50.json",
            "1,2v5-topics-50.json",
            "1,3v5-topics-50.json",
            "2,3v5-topics-50.json",
        ],
    }
    figdict = {
        "Full Term Dataset": "v5-fulldataset.csv",
        "100-Topic Dataset": "v5-topics-100.csv",
        "200-Topic Dataset": "v5-topics-200.csv",
        "400-Topic Dataset": "v5-topics-400.csv",
        "50-Topic Dataset": "v5-topics-50.csv",
    }
    D3list = [
        "v5-fulldataset.json",
        "v5-topics-100.json",
        "v5-topics-200.json",
        "v5-topics-400.json",
        "v5-topics-50.json",
    ]
    listographs = figdict1[tab]
    gimper = {1: [0, 1], 2: [0, 2], 3: [1, 2]}
    dfc = [gimper[2][1]]
    figlist2 = []
    if not gdict == {1: None, 2: None, 3: None, 4: None}:
        import csv

        if not g1c1 == None:

            dicval = list(gdict[1].values())
            dival = dicval[0]
            if "customdata" in dival[0]:
                ddval = dival[0]["customdata"]
            else:
                try:
                    ddval = dival[0]["hovertext"]
                except:
                    ddval = None
            Dframe = {}
            Dframe2 = {}
            Dlist = {}
            with open(join("CSVData", figdict[tab]), newline="") as f:
                reader = csv.reader(f)
                Dlist1 = list(reader)
            for rowl in Dlist1:
                if not rowl == []:
                    Dlist.update({rowl[0]: rowl[1]})
            if "customdata" in dival[0]:
                corr1 = [dival[0]["x"], dival[0]["y"], dival[0]["z"]]
            else:
                try:
                    corr = Dlist[ddval]
                    corr1 = corr.strip("][").split(", ")
                    for id in enumerate(corr1):
                        corr1[id[0]] = float(corr1[id[0]])
                except:
                    corr1 = [dival[0]["x"], dival[0]["y"], dival[0]["z"]]

            for cng in Dlist:
                cng1 = Dlist[cng].strip("][").split(", ")
                for id1 in enumerate(cng1):
                    cng1[id1[0]] = float(cng1[id1[0]])
                dst = distance_finder(corr1, cng1)
                Dframe2.update({cng: dst})
                if dst == 0:
                    continue
                Dframe.update({cng: dst})
            srdict10 = list(
                dict(
                    sorted(Dframe.items(), key=lambda item: item[1], reverse=False)
                ).items()
            )[:10]
            newd = []
            das110 = list(
                dict(sorted(srdict10, key=lambda item: item[1], reverse=True)).items()
            )
            for a in das110:
                b = 1 / a[1]
                newd.append([a[0], b])
            dfn = pd.DataFrame(newd, columns=["Name", "1/Distance"])

            fig = px.bar(
                dfn,
                y="Name",
                x="1/Distance",
                title=None,
                color="1/Distance",
                orientation="h",
                color_continuous_scale="plasma",
                height=900,
            ).update_traces(marker_colorbar_showticklabels=False)

            fig.update_layout(
                title_x=0.5,
                hovermode="closest",
                xaxis=dict(
                    showgrid=False,  # thin lines in the background
                    zeroline=False,  # thick line at x=0
                    title="",
                    visible=False,
                    autorange="reversed"
                    # numbers below
                ),
                yaxis=dict(
                    title="",
                    showgrid=False,  # thin lines in the background
                    zeroline=False,  # thick line at x=0
                ),
            )
            fig = dcc.Graph(figure=fig)
            print(dicval)
            fig1 = listographs[0]
            scale = MinMaxScaler()
            cgrad1 = (
                scale.fit_transform(array(list(Dframe2.values())).reshape(-1, 1))
                .reshape(1, -1)
                .tolist()[0]
            )
            fig2 = Figure(fig.figure)
            Recolor(fig2, cgrad1)
            fig.figure = fig2
            figlist.update({"0": fig})
            figlist2.append(fig1)
            figc = Figure(colfigdict["0"])

            Recolor(figc, cgrad1)
            colfigdict["0"] = figc

        else:
            fig = px.bar(
                title="Click on a point",
                orientation="h",
                color_continuous_scale="plasma_r",
                height=900,
            )

            fig.update_layout(
                title_x=0.5,
                xaxis=dict(
                    showgrid=False,  # thin lines in the background
                    zeroline=False,  # thick line at x=0
                    visible=False,  # numbers below
                ),
                yaxis=dict(
                    showgrid=False,  # thin lines in the background
                    zeroline=False,  # thick line at x=0
                    visible=False,  # numbers below
                ),
            )
            fig = dcc.Graph(figure=fig)
            figlist.update({"0": fig})
            figlist2.append(no_update)

        if not g2c1 == None:
            dicval = list(gdict[2].values())
            dival = dicval[0]
            if "customdata" in dival[0]:
                ddval = dival[0]["customdata"]
            else:
                try:
                    ddval = dival[0]["hovertext"]
                except:
                    ddval = None
            Dframe = {}
            Dframe2 = {}
            Dlist = {}
            with open(join("CSVData", figdict[tab]), newline="") as f:
                reader = csv.reader(f)
                Dlist1 = list(reader)
            for rowl in Dlist1:
                if not rowl == []:
                    Dlist.update({rowl[0]: rowl[1]})
            if "customdata" in dival[0]:
                corr1 = [dival[0]["x"], dival[0]["y"]]
            else:
                try:
                    corr = Dlist[ddval]
                    corr1 = corr.strip("][").split(", ")
                    for id in enumerate(corr1):
                        corr1[id[0]] = float(corr1[id[0]])
                except:
                    corr1 = [dival[0]["x"], dival[0]["y"]]

                for id in enumerate(corr1):
                    corr1[id[0]] = float(corr1[id[0]])
            for cng in Dlist:
                cng1 = Dlist[cng].strip("][").split(", ")
                for id1 in enumerate(cng1):
                    cng1[id1[0]] = float(cng1[id1[0]])
                if len(corr1) >= 3:
                    dst = distance_finder2d(
                        [corr1[gimper[1][0]], corr1[gimper[1][1]]],
                        [cng1[gimper[1][0]], cng1[gimper[1][1]]],
                    )
                elif len(corr1) == 2:
                    dst = distance_finder2d(
                        [corr1[0], corr1[1]], [cng1[gimper[1][0]], cng1[gimper[1][1]]]
                    )
                Dframe2.update({cng: dst})
                if dst == 0:
                    continue
                Dframe.update({cng: dst})
            srdict10 = list(
                dict(
                    sorted(Dframe.items(), key=lambda item: item[1], reverse=False)
                ).items()
            )[:10]
            newd = []
            das110 = list(
                dict(sorted(srdict10, key=lambda item: item[1], reverse=True)).items()
            )
            for a in das110:
                b = 1 / a[1]
                newd.append([a[0], b])
            dfn = pd.DataFrame(newd, columns=["Name", "1/Distance"])

            fig = px.bar(
                dfn,
                y="Name",
                x="1/Distance",
                title=None,
                color="1/Distance",
                orientation="h",
                color_continuous_scale="plasma",
                height=900,
            ).update_traces(marker_colorbar_showticklabels=False)

            fig.update_layout(
                hovermode="closest",
                title_x=0.5,
                xaxis=dict(
                    showgrid=False,  # thin lines in the background
                    zeroline=False,  # thick line at x=0
                    title="",
                    visible=False,
                    autorange="reversed"
                    # numbers below
                ),
                yaxis=dict(
                    title="",
                    showgrid=False,  # thin lines in the background
                    zeroline=False,  # thick line at x=0
                ),
            )
            fig = dcc.Graph(figure=fig)
            print(dicval)
            fig1 = listographs[0]
            scale = MinMaxScaler()
            cgrad1 = (
                scale.fit_transform(array(list(Dframe2.values())).reshape(-1, 1))
                .reshape(1, -1)
                .tolist()[0]
            )
            fig2 = Figure(fig.figure)
            Recolor(fig2, cgrad1)
            fig.figure = fig2
            figlist.update({"1": fig})
            figlist2.append(fig1)
            figc = Figure(colfigdict["1"])

            Recolor(figc, cgrad1)
            figc.update_traces(unselected_marker_opacity=1)
            colfigdict["1"] = figc
        else:
            fig = px.bar(
                title="Click on a point",
                orientation="h",
                color_continuous_scale="plasma_r",
                height=900,
            )

            fig.update_layout(
                title_x=0.5,
                xaxis=dict(
                    showgrid=False,  # thin lines in the background
                    zeroline=False,  # thick line at x=0
                    visible=False,  # numbers below
                ),
                yaxis=dict(
                    showgrid=False,  # thin lines in the background
                    zeroline=False,  # thick line at x=0
                    visible=False,  # numbers below
                ),
            )
            fig = dcc.Graph(figure=fig)
            figlist.update({"1": fig})
        if not g3c1 == None:
            dicval = list(gdict[3].values())
            dival = dicval[0]
            if "customdata" in dival[0]:
                ddval = dival[0]["customdata"]
            else:
                try:
                    ddval = dival[0]["hovertext"]
                except:
                    ddval = None
            Dframe = {}
            Dframe2 = {}
            Dlist = {}
            with open(join("CSVData", figdict[tab]), newline="") as f:
                reader = csv.reader(f)
                Dlist1 = list(reader)
            for rowl in Dlist1:
                if not rowl == []:
                    Dlist.update({rowl[0]: rowl[1]})
            if "customdata" in dival[0]:
                corr1 = [dival[0]["x"], dival[0]["y"]]
            else:
                try:
                    corr = Dlist[ddval]
                    corr1 = corr.strip("][").split(", ")
                    for id in enumerate(corr1):
                        corr1[id[0]] = float(corr1[id[0]])
                except:
                    corr1 = [dival[0]["x"], dival[0]["y"]]
            for cng in Dlist:
                cng1 = Dlist[cng].strip("][").split(", ")
                for id1 in enumerate(cng1):
                    cng1[id1[0]] = float(cng1[id1[0]])
                if len(corr1) >= 3:
                    dst = distance_finder2d(
                        [corr1[gimper[2][0]], corr1[gimper[2][1]]],
                        [cng1[gimper[2][0]], cng1[gimper[2][1]]],
                    )
                elif len(corr1) == 2:
                    dst = distance_finder2d(
                        [corr1[0], corr1[1]], [cng1[gimper[2][0]], cng1[gimper[2][1]]]
                    )
                Dframe2.update({cng: dst})
                if dst == 0:
                    continue
                Dframe.update({cng: dst})
            srdict10 = list(
                dict(
                    sorted(Dframe.items(), key=lambda item: item[1], reverse=False)
                ).items()
            )[:10]
            newd = []
            das110 = list(
                dict(sorted(srdict10, key=lambda item: item[1], reverse=True)).items()
            )
            for a in das110:
                b = 1 / a[1]
                newd.append([a[0], b])
            dfn = pd.DataFrame(newd, columns=["Name", "1/Distance"])

            fig = px.bar(
                dfn,
                y="Name",
                x="1/Distance",
                title=None,
                color="1/Distance",
                orientation="h",
                color_continuous_scale="plasma",
                height=900,
            ).update_traces(marker_colorbar_showticklabels=False)

            fig.update_layout(
                hovermode="closest",
                title_x=0.5,
                xaxis=dict(
                    showgrid=False,  # thin lines in the background
                    zeroline=False,  # thick line at x=0
                    title="",
                    visible=False,
                    autorange="reversed"
                    # numbers below
                ),
                yaxis=dict(
                    title="",
                    showgrid=False,  # thin lines in the background
                    zeroline=False,  # thick line at x=0
                ),
            )
            fig = dcc.Graph(figure=fig)
            print(dicval)
            fig1 = listographs[0]
            scale = MinMaxScaler()
            cgrad1 = (
                scale.fit_transform(array(list(Dframe2.values())).reshape(-1, 1))
                .reshape(1, -1)
                .tolist()[0]
            )
            fig2 = Figure(fig.figure)
            Recolor(fig2, cgrad1)
            fig.figure = fig2
            figlist.update({"2": fig})
            figlist2.append(fig1)
            figc = Figure(colfigdict["2"])

            Recolor(figc, cgrad1)
            figc.update_traces(unselected_marker_opacity=1)
            colfigdict["2"] = figc
        else:
            fig = px.bar(
                title="Click on a point",
                orientation="h",
                color_continuous_scale="plasma_r",
                height=900,
            )

            fig.update_layout(
                title_x=0.5,
                xaxis=dict(
                    showgrid=False,  # thin lines in the background
                    zeroline=False,  # thick line at x=0
                    visible=False,  # numbers below
                ),
                yaxis=dict(
                    showgrid=False,  # thin lines in the background
                    zeroline=False,  # thick line at x=0
                    visible=False,  # numbers below
                ),
            )
            fig = dcc.Graph(figure=fig)
            figlist.update({"2": fig})
        if not g4c1 == None:
            dicval = list(gdict[4].values())
            dival = dicval[0]
            if "customdata" in dival[0]:
                ddval = dival[0]["customdata"]
            else:
                try:
                    ddval = dival[0]["hovertext"]
                except:
                    ddval = None
            Dframe = {}
            Dframe2 = {}
            Dlist = {}
            with open(join("CSVData", figdict[tab]), newline="") as f:
                reader = csv.reader(f)
                Dlist1 = list(reader)
            for rowl in Dlist1:
                if not rowl == []:
                    Dlist.update({rowl[0]: rowl[1]})
            if "customdata" in dival[0]:
                corr1 = [dival[0]["x"], dival[0]["y"]]
            else:
                try:
                    corr = Dlist[ddval]
                    corr1 = corr.strip("][").split(", ")
                    for id in enumerate(corr1):
                        corr1[id[0]] = float(corr1[id[0]])
                except:
                    corr1 = [dival[0]["x"], dival[0]["y"]]
            for cng in Dlist:
                cng1 = Dlist[cng].strip("][").split(", ")
                for id1 in enumerate(cng1):
                    cng1[id1[0]] = float(cng1[id1[0]])
                if len(corr1) >= 3:
                    dst = distance_finder2d(
                        [corr1[gimper[3][0]], corr1[gimper[3][1]]],
                        [cng1[gimper[3][0]], cng1[gimper[3][1]]],
                    )
                elif len(corr1) == 2:
                    dst = distance_finder2d(
                        [corr1[0], corr1[1]], [cng1[gimper[3][0]], cng1[gimper[3][1]]]
                    )
                Dframe2.update({cng: dst})
                if dst == 0:
                    continue
                Dframe.update({cng: dst})
            srdict10 = list(
                dict(
                    sorted(Dframe.items(), key=lambda item: item[1], reverse=False)
                ).items()
            )[:10]
            newd = []
            das110 = list(
                dict(sorted(srdict10, key=lambda item: item[1], reverse=True)).items()
            )
            for a in das110:
                b = 1 / a[1]
                newd.append([a[0], b])
            dfn = pd.DataFrame(newd, columns=["Name", "1/Distance"])

            fig = px.bar(
                dfn,
                y="Name",
                x="1/Distance",
                title=None,
                color="1/Distance",
                orientation="h",
                color_continuous_scale="plasma",
                height=900,
            ).update_traces(marker_colorbar_showticklabels=False)

            fig.update_layout(
                hovermode="closest",
                title_x=0.5,
                xaxis=dict(
                    showgrid=False,  # thin lines in the background
                    zeroline=False,  # thick line at x=0
                    title="",
                    visible=False,
                    autorange="reversed"
                    # numbers below
                ),
                yaxis=dict(
                    title="",
                    showgrid=False,  # thin lines in the background
                    zeroline=False,  # thick line at x=0
                ),
            )
            fig = dcc.Graph(figure=fig)
            print(dicval)
            fig1 = listographs[0]
            scale = MinMaxScaler()
            cgrad1 = (
                scale.fit_transform(array(list(Dframe2.values())).reshape(-1, 1))
                .reshape(1, -1)
                .tolist()[0]
            )
            fig2 = Figure(fig.figure)
            Recolor(fig2, cgrad1)
            fig.figure = fig2
            figlist.update({"3": fig})
            figlist2.append(fig1)
            figc = Figure(colfigdict["3"])

            Recolor(figc, cgrad1)
            figc.update_traces(unselected_marker_opacity=1)
            colfigdict["3"] = figc
        else:
            fig = px.bar(
                title="Click on a point",
                orientation="h",
                color_continuous_scale="plasma_r",
                height=900,
            )

            fig.update_layout(
                title_x=0.5,
                xaxis=dict(
                    showgrid=False,  # thin lines in the background
                    zeroline=False,  # thick line at x=0
                    visible=False,  # numbers below
                ),
                yaxis=dict(
                    showgrid=False,  # thin lines in the background
                    zeroline=False,  # thick line at x=0
                    visible=False,  # numbers below
                ),
            )
            fig = dcc.Graph(figure=fig)
            figlist.update({"3": fig})
        return (
            figlist["0"],
            figlist["1"],
            figlist["2"],
            figlist["3"],
            no_update,
            no_update,
            no_update,
            no_update,
            colfigdict["0"],
            colfigdict["1"],
            colfigdict["2"],
            colfigdict["3"],
        )
    else:
        return (
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
        )


@app.callback(
    Output("session", "data"),
    Output("button", "n_clicks"),
    Output("loading-output", "children"),
    [Input("upload-data", "filename"), Input("upload-data", "contents")],
)
def save_file(name, content):
    if not [name, content] == [None, None]:
        SaveandEncode(name, content)
        return join(UPLOAD_DIRECTORY, name[0]), no_update, f"Upload Complete"
    else:
        return (
            None,
            None,
            f"Drag and drop or select a .nii.gz or .zip archive containing fMRI data to begin",
        )


def SaveandEncode(name, content):
    with open(join(UPLOAD_DIRECTORY, name[0]), "wb") as fp:
        fp.write(decodebytes(content[0].encode("utf8").split(b";base64,")[1]))


@app.callback(
    Output("session2", "data"),
    Output("session3", "data"),
    Output("session4", "data"),
    Output("loading-output2", "children"),
    Output("iszipped", "data"),
    Output("button", "disabled"),
    Input("session", "data"),
    prevent_initial_callbacks=True,
)
def analyze(file):
    Disp = False
    if not file == None:
        if exists(file) == True:
            if file.rsplit(".")[-1] == "zip":
                distancedict = {}
                dicts1 = {}
                dicts2 = {"Map Name": ["Gradient 1", "Gradient 2", "Gradient 3"]}
                delif("extTemp")
                ad = []
                filelistd = []
                with ZipFile(file, "r") as zip_ref:
                    zip_ref.extractall("extTemp")
                remove(file)
                GradientList = []
                WordList = []
                for subdir, dirs, files in walk("extTemp"):
                    for file in files:
                        filepath = subdir + sep + file
                        if filepath.endswith(".nii.gz"):
                            fp = (filepath.split("\\", 2)[-1]).split(".nii")[0]
                            filelistd.append(filepath)
                            print(fp)
                gslist = []
                for gs in listdir(dirname(abspath(__file__)) + "//Gradients"):
                    gslist.append(gs)
                    mask, WordList, GradientList = maskingcalc(
                        WordList, GradientList, gs
                    )
                GradientFrame = [GradientList[0], GradientList[1], GradientList[2]]
                for zzz in filelistd:
                    Corrs1 = []
                    WordList = []

                    maskedSeries = apply_mask(
                        resample_to_img(
                            niload(zzz),
                            niload(
                                join(dirname(abspath(__file__)) + "//Gradients", gs)
                            ),
                        ),
                        mask,
                    )
                    min_max_scaler = preprocessing.MinMaxScaler()
                    np_scaled = squeeze(
                        min_max_scaler.fit_transform(
                            maskedSeries.astype("float64").reshape(-1, 1)
                        )
                    )
                    min_max_scaler1 = preprocessing.MinMaxScaler()
                    np_scaled1 = squeeze(
                        min_max_scaler1.fit_transform(
                            GradientFrame[0].astype("float64").reshape(-1, 1)
                        )
                    )
                    min_max_scaler2 = preprocessing.MinMaxScaler()
                    np_scaled2 = squeeze(
                        min_max_scaler2.fit_transform(
                            GradientFrame[1].astype("float64").reshape(-1, 1)
                        )
                    )
                    min_max_scaler3 = preprocessing.MinMaxScaler()
                    np_scaled3 = squeeze(
                        min_max_scaler3.fit_transform(
                            GradientFrame[2].astype("float64").reshape(-1, 1)
                        )
                    )
                    Corrs = [
                        pearsonr(np_scaled1, np_scaled)[0],
                        pearsonr(np_scaled2, np_scaled)[0],
                        pearsonr(np_scaled3, np_scaled)[0],
                    ]

                    # def msk(a,b,mskr):
                    #     maskf = NiftiMasker()
                    #     maskf.fit(mskr)
                    #     #a = apply_mask(a,mskr)
                    #     #b = apply_mask(b,mskr)
                    #     kwargs = {"ref_imgs": [a], "src_imgs": [b], "masker": maskf}
                    #     return kwargs

                    # Corrs = [
                    #         plot_img_comparison(**msk(resample_to_img(niload(zzz), niload(join(dirname(abspath(__file__)) + "//Gradients", gslist[0]))), niload(join(dirname(abspath(__file__)) + "//Gradients", gslist[0])),mask))[0],
                    #         plot_img_comparison(**msk(resample_to_img(niload(zzz), niload(join(dirname(abspath(__file__)) + "//Gradients", gslist[1]))), niload(join(dirname(abspath(__file__)) + "//Gradients", gslist[1])),mask))[0],
                    #         plot_img_comparison(**msk(resample_to_img(niload(zzz), niload(join(dirname(abspath(__file__)) + "//Gradients", gslist[2]))), niload(join(dirname(abspath(__file__)) + "//Gradients", gslist[2])),mask))[0]

                    #     ]
                    for i in listdir(dirname(abspath(__file__)) + "//CSVData"):
                        Alldistanc, top10, integer, pv_in_hex = analyzefunc(Corrs, i)
                        distancedict.update({i: Alldistanc})
                    print("done")
                    Disp = True
                    dicts1.update({zzz: Corrs})
                    dicts2.update(
                        {
                            zzz.split("\\")[-1].split(".")[0]: [
                                Corrs[0],
                                Corrs[1],
                                Corrs[2],
                            ]
                        }
                    )
                remove(zzz)
                iszip = True

                with open("Coords.csv", "w", newline="") as csv_file:
                    writer = csv.writer(csv_file)
                    for key, value in dicts2.items():
                        writer.writerow([key, value[0], value[1], value[2]])

                return (
                    distancedict,
                    dicts1,
                    Disp,
                    f"File has been processed, click here to re-render the figures",
                    iszip,
                    False,
                )
            elif file.rsplit(".", 2)[0] or file.rsplit(".", 2)[1] == ".nii.gz":
                if exists(file) == True:
                    Disp = False
                    ad = []

                    if exists(file) == True:
                        Corrs1 = []
                        WordList = []
                        GradientList = []
                        gslist = []
                        for gs in listdir("Gradients"):
                            gslist.append(gs)
                            mask, WordList, GradientList = maskingcalc(
                                WordList, GradientList, gs
                            )
                        GradientFrame = [
                            GradientList[0],
                            GradientList[1],
                            GradientList[2],
                        ]

                        maskedSeries = apply_mask(
                            resample_to_img(
                                niload(file), niload(join("Gradients", gs))
                            ),
                            mask,
                        )
                        min_max_scaler = preprocessing.MinMaxScaler()
                        np_scaled = min_max_scaler.fit_transform(
                            maskedSeries.astype("float64").reshape(-1, 1)
                        )
                        min_max_scaler1 = preprocessing.MinMaxScaler()
                        np_scaled1 = min_max_scaler1.fit_transform(
                            GradientFrame[0].astype("float64").reshape(-1, 1)
                        )
                        min_max_scaler2 = preprocessing.MinMaxScaler()
                        np_scaled2 = min_max_scaler2.fit_transform(
                            GradientFrame[1].astype("float64").reshape(-1, 1)
                        )
                        min_max_scaler3 = preprocessing.MinMaxScaler()
                        np_scaled3 = min_max_scaler3.fit_transform(
                            GradientFrame[2].astype("float64").reshape(-1, 1)
                        )
                        Corrs = [
                            pearsonr(np_scaled1, np_scaled)[0],
                            pearsonr(np_scaled2, np_scaled)[0],
                            pearsonr(np_scaled3, np_scaled)[0],
                        ]
                        # Corrs = [
                        #     plot_img_comparison([resample_to_img(niload(file), niload(join("Gradients", gslist[0])))], [niload(join("Gradients", gslist[0]))],masker=mask)[0],
                        #     plot_img_comparison([resample_to_img(niload(file), niload(join("Gradients", gslist[1])))], [niload(join("Gradients", gslist[1]))],masker=mask)[0],
                        #     plot_img_comparison([resample_to_img(niload(file), niload(join("Gradients", gslist[2])))], [niload(join("Gradients", gslist[2]))],masker=mask)[0]

                        # ]
                        remove(file)

                        for i in listdir("CSVData"):
                            Alldistanc, top10, integer, pv_in_hex = analyzefunc(
                                Corrs, i
                            )
                            colors_hex = squeeze(array(pv_in_hex).T)

                            colour_dict = dict(zip(top10[0], colors_hex))
                            freq_dict = dict(zip(top10[0], integer))

                            def color_func(
                                word, *args, **kwargs
                            ):  # colour function to supply to wordcloud function.. don't ask !
                                try:
                                    color = colour_dict[word]
                                except KeyError:
                                    color = "#000000"  # black
                                return color

                            wc = WordCloud(
                                background_color="white",
                                color_func=color_func,
                                width=400,
                                height=400,
                                prefer_horizontal=1,
                                min_font_size=8,
                                max_font_size=200,
                            )
                            wc = wc.generate_from_frequencies(frequencies=freq_dict)
                            wc.to_file(
                                join("assets", "wordcloud%s.png" % (i.split(".")[0]))
                            )
                            arrco = Corrs
                            Corrs1.append(list(arrco))
                            ad.append(Alldistanc)
                        print("done")
                        Disp = True
                        distancedict = {
                            bbb: ad[em] for em, bbb in enumerate(listdir("CSVData"))
                        }
                        iszip = False
                        file1 = Linuxfix(file, -1)
                        Corrs = {file1: Corrs}
                        # file = None

                        return (
                            distancedict,
                            Corrs,
                            Disp,
                            f"Files have been processed, click here to re-render the figures",
                            iszip,
                            False,
                        )
                    else:
                        return (
                            no_update,
                            no_update,
                            Disp,
                            no_update,
                            no_update,
                            no_update,
                        )
                else:
                    return no_update, no_update, Disp, no_update, no_update, no_update
            else:
                return no_update, no_update, Disp, no_update, no_update, no_update
        else:
            return no_update, no_update, Disp, no_update, no_update, no_update
    else:
        return no_update, no_update, Disp, no_update, no_update, no_update


def Linuxfix(file, x):
    if "/" in file:
        file = file.split(".")[0].split("/")[x]
    elif "\\" in file:
        file = file.split(".")[0].split("\\")[x]
    return file


def takeSecond(elem):
    return elem[1]


def Extract(lst, num):
    return [item[num] for item in lst]


def Extract2(lst, q):
    q = q - 1
    it = [(itenum, item) for itenum, item in enumerate(lst) if itenum <= q]
    it = Extract(it, 1)
    return it


def analyzefunc(Corrs, i):
    import csv

    Dframe = []
    with open(join(dirname(abspath(__file__)) + "//CSVData", i), newline="") as f:
        reader = csv.reader(f)
        Dlist = list(reader)
    for rowl in Dlist:
        if not rowl == []:
            Dframe.append(rowl)
    """    Dframe = pd.read_csv(
        os.path.join("CSVData", i),
        header=None,
        names=["Term", "Coordinates"],
    ).set_index("Term")"""
    # dfdists = pd.DataFrame(columns=["Term", "Distance"])
    dfdists = []

    for ind, v in enumerate(Dframe):
        rowval = v[1].strip("][").split(", ")
        for id in enumerate(rowval):
            rowval[id[0]] = float(rowval[id[0]])
        # distance = distance_finder(Corrs, rowval)
        listst = [v[0], rowval]
        dfdists.append(listst)
        listst = []
    Alldistanc = [x[1] for x in dfdists]
    dfsorted = sorted(dfdists, key=takeSecond, reverse=True)
    scaler = StandardScaler()
    test = array(Extract(dfsorted, 1)).reshape(-1, 1)
    dfsorted1 = [Extract(dfsorted, 0), scaler.fit_transform(test).tolist()]
    top10 = [Extract2(dfsorted1[0], 10), Extract2(dfsorted1[1], 10)]

    # Dictver = np.array(top10["Distance"])
    # Dictver = dict(enumerate(Dictver.flatten(), 1))

    df = array(top10[1])
    absolute = abs(df)  # make absolute
    integer = 100 * absolute  # make interger
    integer = squeeze(integer.astype(int))
    principle_vector = array(df, dtype=float).reshape(-1, 1)  # turn df into array
    pv_in_hex = []
    # get the maximum absolute value in array
    vmax = abs(principle_vector).max()
    vmin = -vmax  # minimu
    # loop through each column (cap)
    for g in range(principle_vector.shape[1]):
        rescale = (principle_vector[:, g] - vmin) / (vmax - vmin)  # rescale scores
        colors_hex = []
        for c in RdBu_r(rescale):
            colors_hex.append(to_hex(c))  # adds colour codes (hex) to list
        pv_in_hex.append(colors_hex)
        # add all colour codes for each item on all caps
    return Alldistanc, top10, integer, pv_in_hex


def maskingcalc(WordList, GradientList, gs):
    WordList.append(gs)
    mask = compute_background_mask(join(dirname(abspath(__file__)) + "//Gradients", gs))
    GradientList.append(
        apply_mask(join(dirname(abspath(__file__)) + "//Gradients", gs), mask)
    )
    return mask, WordList, GradientList


def delif(x):
    if exists(x) == True:
        rmtree(x)
    if not (exists(x) == True):
        mkdir(x)


def GenGraphsInit(figwanted, dir=dirname(abspath(__file__)) + "//Fig_Jsons"):
    empty = True
    if empty == True:
        dir = dirname(abspath(__file__)) + "//Empty_Fig_Jsons"
    fighhh = []
    for anum, a in enumerate(figwanted):
        fig = read_json(join(dir, a))
        figdict2 = {
            "v5-fulldataset.json": "All neurosynth terms",
            "1,2v5-fulldataset.json": "Gradient 1 and 2 against all neurosynth terms",
            "1,3v5-fulldataset.json": "Gradient 1 and 3 against all neurosynth terms",
            "2,3v5-fulldataset.json": "Gradient 2 and 3 against all neurosynth terms",
            "v5-topics-100.json": "100 neurosynth topics",
            "1,2v5-topics-100.json": "Gradient 1 and 2 against 100 neurosynth topics",
            "1,3v5-topics-100.json": "Gradient 1 and 3 against 100 neurosynth topics",
            "2,3v5-topics-100.json": "Gradient 2 and 3 against 100 neurosynth topics",
            "v5-topics-200.json": "200 neurosynth topics",
            "1,2v5-topics-200.json": "Gradient 1 and 2 against 200 neurosynth topics",
            "1,3v5-topics-200.json": "Gradient 1 and 3 against 200 neurosynth topics",
            "2,3v5-topics-200.json": "Gradient 2 and 3 against 200 neurosynth topics",
            "v5-topics-400.json": "400 neurosynth topics",
            "1,2v5-topics-400.json": "Gradient 1 and 2 against 400 neurosynth topics",
            "1,3v5-topics-400.json": "Gradient 1 and 3 against 400 neurosynth topics",
            "2,3v5-topics-400.json": "Gradient 2 and 3 against 400 neurosynth topics",
            "v5-topics-50.json": "50 neurosynth topics",
            "1,2v5-topics-50.json": "Gradient 1 and 2 against 50 neurosynth topics",
            "1,3v5-topics-50.json": "Gradient 1 and 3 against 50 neurosynth topics",
            "2,3v5-topics-50.json": "Gradient 2 and 3 against 50 neurosynth topics",
        }
        if empty == True:
            fig.update_layout(
                title_x=0.5,
                autosize=True,
                hovermode="closest",
                # width=900,
                clickmode="event+select",
                height=900,
                # template="simple_white",
            )
        else:
            fig.update_layout(
                title={"text": figdict2[a], "xanchor": "center"},
                title_x=0.5,
                autosize=True,
                hovermode="closest",
                # width=900,
                clickmode="event+select",
                height=900,
                # template="simple_white",
            )
        # fig.update_traces(marker_text="OGp")
        fig = dcc.Graph(
            config={"displaylogo": False},
            id=("graph" + str(anum + 1) + "col1"),
            figure=fig,
            style={"textAlign": "center"},
        )
        fighhh.append(fig)
        print("ooo")
    return fighhh


@app.callback(
    [Output("store1", "data"),],
    Output("geng", "data"),
    Output("cl", "data"),
    Output("loading-output3", "children"),
    Input("button", "n_clicks"),
    Input("session4", "data"),
    Input("session3", "data"),
    Input("session2", "data"),
    Input("tabs", "active_tab"),
    Input("iszipped", "data"),
)
def generate_graphs(n, ismade, data2, distances, input, iszip):
    savefigs = True
    nopoints = True
    cl1 = None
    disornot = None
    # figl = listdir("Fig_Jsons")
    figdict = {
        "Full Term Dataset": [
            "v5-fulldataset.json",
            "1,2v5-fulldataset.json",
            "1,3v5-fulldataset.json",
            "2,3v5-fulldataset.json",
        ],
        "100-Topic Dataset": [
            "v5-topics-100.json",
            "1,2v5-topics-100.json",
            "1,3v5-topics-100.json",
            "2,3v5-topics-100.json",
        ],
        "200-Topic Dataset": [
            "v5-topics-200.json",
            "1,2v5-topics-200.json",
            "1,3v5-topics-200.json",
            "2,3v5-topics-200.json",
        ],
        "400-Topic Dataset": [
            "v5-topics-400.json",
            "1,2v5-topics-400.json",
            "1,3v5-topics-400.json",
            "2,3v5-topics-400.json",
        ],
        "50-Topic Dataset": [
            "v5-topics-50.json",
            "1,2v5-topics-50.json",
            "1,3v5-topics-50.json",
            "2,3v5-topics-50.json",
        ],
    }

    D3list = [
        "v5-fulldataset.json",
        "v5-topics-100.json",
        "v5-topics-200.json",
        "v5-topics-400.json",
        "v5-topics-50.json",
    ]
    colis = qualitative.Dark24

    if not n:
        listographs = GenGraphsInit(figdict[input])
        ismade, cl1, disornot, listv = PageHandler(listographs)

        return listv, disornot, cl1, f""
    if ismade == True:
        listographs = GenGraphsInit(figdict[input])
        cl1 = True
        disornot = 1
        figlist = []
        cgradient = {}
        if iszip == False or None:
            enu = figdict[input]
            listodis = []
            scale2 = MinMaxScaler()
            for currinp in enu:
                if currinp == enu[0]:
                    for elmne in distances[enu[0].split(".")[0] + ".csv"]:
                        cgrad1 = []

                        disty = distance_finder(list(data2.values())[0], elmne)
                        listodis.append(disty)
                    cgrad1 = (
                        scale2.fit_transform(array(listodis).reshape(-1, 1))
                        .reshape(1, -1)
                        .tolist()[0]
                    )
                    cgradient.update({currinp: cgrad1})
                    listodis = []
                else:
                    f = [
                        int(currinp.split("v")[0].split(",")[0]) - 1,
                        int(currinp.split("v")[0].split(",")[1]) - 1,
                    ]
                    for elmne in distances[enu[0].split(".")[0] + ".csv"]:
                        cgrad1 = []
                        l = list(data2.values())
                        disty = distance_finder2d(
                            [
                                list(data2.values())[0][f[0]],
                                list(data2.values())[0][f[1]],
                            ],
                            [elmne[f[0]], elmne[f[1]]],
                        )
                        listodis.append(disty)
                    cgrad1 = (
                        scale2.fit_transform(array(listodis).reshape(-1, 1))
                        .reshape(1, -1)
                        .tolist()[0]
                    )
                    cgradient.update({currinp: cgrad1})
                    listodis = []
            for benu, abc in enumerate(enu):
                if abc in D3list:

                    fig = listographs[benu].figure
                    Recolor(fig, cgradient[abc])
                    abc = abc.split(".")[0] + ".csv"
                    ref = Scatter3d(
                        x=[list(data2.values())[0][0]],
                        y=[list(data2.values())[0][1]],
                        z=[list(data2.values())[0][2]],
                        mode="markers+text",
                        customdata=[Linuxfix(list(data2.keys())[0], -1)],
                        name=list(data2.keys())[0],
                        text=list(data2.keys())[0],
                        hoverinfo=["text"],
                        hovertext=list(data2.keys())[0],
                        marker=dict(
                            size=12, color="blue", symbol="diamond", opacity=0.95,
                        ),
                    )

                    fig.add_trace(ref)
                    fig.update_layout(
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1,
                        )
                    )
                    fig = dcc.Graph(
                        config={"displaylogo": False},
                        id="graph1col1",
                        figure=fig,
                        style={"textAlign": "center"},
                    )
                    figlist = [fig]
                else:
                    u = [
                        int(abc.split("v")[0].split(",")[0]) - 1,
                        int(abc.split("v")[0].split(",")[1]) - 1,
                    ]
                    fig = listographs[benu].figure
                    Recolor(fig, cgradient[abc])
                    ref = Scatter(
                        #
                        x=[list(data2.values())[0][u[0]]],
                        y=[list(data2.values())[0][u[1]]],
                        mode="markers+text",
                        text=list(data2.keys())[0],
                        name=list(data2.keys())[0],
                        hovertext=list(data2.keys())[0],
                        hoverinfo="text",
                        customdata=[Linuxfix(list(data2.keys())[0], -1)],
                        marker=dict(
                            size=12, color="blue", symbol="diamond", opacity=0.95,
                        ),
                    )

                    fig.add_trace(ref)
                    fig.update_layout(
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1,
                        )
                    )
                    fig = dcc.Graph(
                        config={"displaylogo": False},
                        id="graph" + str(benu + 1) + "col1",
                        figure=fig,
                        style={"textAlign": "center"},
                    )
                    figlist.append(fig)
                if input == "50-Topic Dataset":
                    cl1 = app.get_asset_url("wordcloudv5-topics-50.png")
                elif input == "100-Topic Dataset":
                    cl1 = app.get_asset_url("wordcloudv5-topics-100.png")
                elif input == "200-Topic Dataset":
                    cl1 = app.get_asset_url("wordcloudv5-topics-200.png")
                elif input == "400-Topic Dataset":
                    cl1 = app.get_asset_url("wordcloudv5-topics-400.png")
                elif input == "Full Term Dataset":
                    cl1 = app.get_asset_url("wordcloudv5-fulldataset.png")

            return figlist, disornot, cl1, f""
        elif iszip == True:
            enu = figdict[input]
            disters = []
            listodis = []
            scale1 = MinMaxScaler()
            for isd, cdc in enumerate(distances):
                bbbbbb = distances[cdc]
                disbo = []
                for bb in bbbbbb:
                    listodis = []
                    for vvrs in data2:
                        dstn = distance_finder(bb, data2[vvrs])
                        listodis.append(dstn)
                    disbo.append(listodis)
                disters.append(disbo)
            cgradient = []
            for elmne in disters:
                cgrad1 = []
                for eld in elmne:
                    cgrad1.append(sum(eld) / len(eld))
                cgrad1 = (
                    scale1.fit_transform(array(cgrad1).reshape(-1, 1))
                    .reshape(1, -1)
                    .tolist()[0]
                )
                cgradient.append(cgrad1)

            graddict = {
                "Full Term Dataset": cgradient[0],
                "100-Topic Dataset": cgradient[1],
                "200-Topic Dataset": cgradient[2],
                "400-Topic Dataset": cgradient[3],
                "50-Topic Dataset": cgradient[4],
            }
            bruhlist = []
            for odr in enu:
                if not odr[0] == "v":
                    f = [
                        int(odr.split("v")[0].split(",")[0]) - 1,
                        int(odr.split("v")[0].split(",")[1]) - 1,
                    ]
                    curdlist = []
                    for abxc in distances[figdict[input][0].split(".")[0] + ".csv"]:
                        cordlist = []
                        for labell in data2:
                            cord = data2[labell][f[0]], data2[labell][f[1]]
                            curd = abxc[f[0]], abxc[f[1]]
                            dsc = distance_finder2d(cord, curd)
                            cordlist.append(dsc)
                        avge = sum(cordlist) / len(cordlist)
                        curdlist.append(avge)
                    curdlist = (
                        scale1.fit_transform(array(curdlist).reshape(-1, 1))
                        .reshape(1, -1)
                        .tolist()[0]
                    )
                    bruhlist.append(curdlist)
            bruhdict = {
                "1,2": bruhlist[0],
                "1,3": bruhlist[1],
                "2,3": bruhlist[2],
            }
            grplist = []
            numchek = 0
            grpdict = {}
            for benu, abc in enumerate(enu):

                fig = listographs[benu].figure

                if abc in D3list:
                    Recolor(fig, graddict[input])
                    for ert, dn in enumerate(data2):
                        grpnm = Linuxfix(dn, -2)
                        # grpnm = dn.split("\\")[-2]
                        if not grpnm in grplist:
                            grplist.append(grpnm)
                            grpdict.update({grpnm: colis[numchek]})
                            numchek = numchek + 1
                        abc = abc.split(".")[0] + ".csv"
                        ref = Scatter3d(
                            x=[data2[dn][0]],
                            y=[data2[dn][1]],
                            z=[data2[dn][2]],
                            legendgroup=grpnm,
                            legendgrouptitle_text=grpnm,
                            mode="markers",
                            name=Linuxfix(dn, -1),
                            hoverinfo="name",
                            # customdata=[Linuxfix(list(data2.keys())[0], -1)],
                            marker=dict(
                                size=12,
                                color=grpdict[grpnm],
                                symbol="diamond",
                                opacity=0.95,
                            ),
                        )

                        fig.add_trace(ref)
                        # fig.update_layout(
                        #     legend=dict(
                        #         orientation="h",
                        #         yanchor="bottom",
                        #         y=1.02,
                        #         xanchor="right",
                        #         x=1)
                        # )
                    fig.update_xaxes(
                        showgrid=True,
                        zeroline=True,
                        gridcolor="Grey",
                        zerolinecolor="Grey",
                    )
                    fig.update_yaxes(
                        showgrid=True,
                        zeroline=True,
                        gridcolor="Grey",
                        zerolinecolor="Grey",
                    )
                    if savefigs == True:
                        fig.write_html(
                            dirname(abspath(__file__))
                            + "//Figure_Output//"
                            + abc.split(".")[0]
                            + ".html"
                        )
                        fig.write_image(
                            dirname(abspath(__file__))
                            + "//Figure_Output//"
                            + abc.split(".")[0]
                            + ".jpeg"
                        )

                    # fig.update_zaxes(gridcolor="Grey", overwrite=True, showgrid=True)

                    fig = dcc.Graph(
                        config={"displaylogo": False},
                        id="graph" + str(benu + 1) + "col1",
                        figure=fig,
                        style={"textAlign": "center"},
                    )
                    figlist.append(fig)
                else:
                    for est, dn in enumerate(data2):
                        grpnm = Linuxfix(dn, -2)
                        est = est + ert
                        fig = listographs[benu].figure
                        Recolor(fig, bruhdict[abc.split("v")[0]])

                        u = [
                            (int(abc.split("v")[0].split(",")[0]) - 1),
                            (int(abc.split("v")[0].split(",")[1]) - 1),
                        ]
                        ref1 = Scatter(
                            #
                            x=[data2[dn][u[0]]],
                            y=[data2[dn][u[1]]],
                            name=Linuxfix(dn, -1),
                            legendgroup=grpnm,
                            legendgrouptitle_text=grpnm,
                            mode="markers",
                            text=Linuxfix(dn, -1),
                            textposition="top center",
                            textfont={"size": 30},
                            hoverinfo="name",
                            # customdata=[Linuxfix(list(data2.keys())[0], -1)],
                            marker=dict(
                                size=30,
                                color=grpdict[grpnm],
                                symbol="diamond",
                                opacity=0.95,
                            ),
                        )
                        # fig.data[0]._parent.data[1].textfont.size = 20
                        fig.add_trace(ref1)
                        fig.update_xaxes(
                            ticktext=["Unimodal", "Transmodal"],
                            # tickvals=[-0.6, 0.6],
                        )
                        fig.update_yaxes(
                            ticktext=["Task-positive", "Task-negative"],
                            # tickvals=[-0.6, 0.6],
                        )
                        fig.update_layout(
                            showlegend=False,
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1,
                            ),
                        )
                    fig.update_xaxes(
                        tickfont_size=30,
                        showgrid=True,
                        zeroline=True,
                        gridcolor="LightGrey",
                        zerolinecolor="LightGrey",
                        range=[-0.6, 0.6],
                    )
                    fig.update_yaxes(
                        tickfont_size=30,
                        showgrid=True,
                        zeroline=True,
                        gridcolor="LightGrey",
                        zerolinecolor="LightGrey",
                        range=[-0.6, 0.6],
                    )
                    fig.update_layout(
                        xaxis_title_font={"size": 40},
                        yaxis_title_font={"size": 40},
                        showlegend=False,
                    )
                    if savefigs == True:
                        fig.write_html(
                            dirname(abspath(__file__))
                            + "//Figure_Output//"
                            + abc.split(".")[0]
                            + ".html"
                        )
                        fig.write_image(
                            dirname(abspath(__file__))
                            + "//Figure_Output//"
                            + abc.split(".")[0]
                            + ".jpeg"
                        )

                    fig = dcc.Graph(
                        config={"displaylogo": False},
                        id="graph" + str(benu + 1) + "col1",
                        figure=fig,
                        style={"textAlign": "center"},
                    )

                    figlist.append(fig)
                if input == "50-Topic Dataset":
                    cl1 = app.get_asset_url("wordcloudv5-topics-50.png")
                elif input == "100-Topic Dataset":
                    cl1 = app.get_asset_url("wordcloudv5-topics-100.png")
                elif input == "200-Topic Dataset":
                    cl1 = app.get_asset_url("wordcloudv5-topics-200.png")
                elif input == "400-Topic Dataset":
                    cl1 = app.get_asset_url("wordcloudv5-topics-400.png")
                elif input == "Full Term Dataset":
                    cl1 = app.get_asset_url("wordcloudv5-fulldataset.png")
                disornot = False
            return figlist, disornot, cl1, f""
        else:
            return no_update, no_update, no_update, no_update
    else:
        return no_update, no_update, no_update, no_update


def Recolor(figure, cgradient):
    figure.update_traces(
        overwrite=True,
        marker_color=cgradient,
        marker_colorscale="thermal",
        marker_reversescale=True,
        selector=dict(marker_symbol="circle"),
    )


def distance_finder(one, two):
    [x1, y1, z1] = one  # first coordinates
    [x2, y2, z2] = two  # second coordinates
    return (((x2 - x1) ** 2) + ((y2 - y1) ** 2) + ((z2 - z1) ** 2)) ** (1 / 2)


def distance_finder2d(one, two):
    [x1, y1] = one
    [x2, y2] = two
    return (((x2 - x1) ** 2) + ((y2 - y1) ** 2)) ** (1 / 2)


def PageHandler(listv):
    disornot = 0
    cl1 = None
    ismade = False
    return ismade, cl1, disornot, listv


# webbrowser.open('http://127.0.0.1:8888/')
if __name__ == "__main__":
    app.run_server(debug=False)
