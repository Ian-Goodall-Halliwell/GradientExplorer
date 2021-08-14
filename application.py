from dash.dash import no_update
from flask import Flask
from os.path import exists, join
from os import makedirs, mkdir
from dash_bootstrap_components.themes import COSMO
from flask import send_from_directory
from dash import Dash
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
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
from plotly.graph_objects import Scatter, Scatter3d
from plotly.colors import qualitative


UPLOAD_DIRECTORY = "Downloads"
if not exists(UPLOAD_DIRECTORY):
    makedirs(UPLOAD_DIRECTORY)

server = Flask(__name__)
app = Dash(
    server=server,
    external_stylesheets=[COSMO],
    suppress_callback_exceptions=True,
    include_assets_files=False,
    title="Gradient Maker",
    update_title="Processing..."
)
application = app.server


@server.before_first_request
def before_first_request():
    delif("assets")


@server.route('/favicon.ico')
def favicon():
    return send_from_directory('static',
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')


@server.route("/download/<path:path>")
def download(path):
    """Serve a file from the upload directory."""
    return send_from_directory(UPLOAD_DIRECTORY, path, as_attachment=True)


app.layout = dbc.Container(
    [
        dbc.Spinner([html.Div([""], id="loading-output4"),
                     html.Div([""], id="loading-output3")],
                    fullscreen=True,
                    ),
        html.H1("The Gradient-inator"),
        html.H5(
            "Upload an fMRI file in a nibabel-compatible format and wait ~60s for the file to be processed. When the processing is done, click the blue button below. Also I don't really understand how browser data caching works so if you refresh the page while processing data it may break."
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
            children=dbc.Spinner(html.Div(
                ["Drag and drop or click to select a scan to upload."],
                id="loading-output",
            )
            ),
        ),
        dcc.Store(id="session", storage_type="local"),
        dcc.Store(id="session2", storage_type="local"),
        dcc.Store(id="session3", storage_type="local"),
        dcc.Store(id="session4", storage_type="local"),
        dcc.Store(id="geng", storage_type="session"),
        dcc.Store(id="store1", storage_type="session"),
        dcc.Store(id="cl", storage_type="session"),
        dcc.Store(id="iszipped", storage_type="session"),
        html.Hr(),
        dbc.Button(
            color="primary",
            block=True,
            id="button",
            className="mb-3",
            children=dbc.Spinner(
                html.Div(
                    ["Place scan into gradient space"],
                    id="loading-output2",
                )
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
        html.Div(
            id="tab-content",
            className="p-4",
        ),
    ]
)


@app.callback(
    Output("loading-output4", "children"),
    Output("tab-content", "children"),
    Input("tabs", "active_tab"),
    [
        Input("store1", "data"),
    ],
    Input("geng", "data"),
    Input("cl", "data"),
)
def render_tab_content(active_tab, data1, displayoptions, cldata):
    if active_tab and data1 is not None:
        if active_tab == "50-Topic Dataset":
            return f"", dbc.Row(
                [
                    dbc.Col(
                        [
                            dcc.Graph(
                                figure=data1[0],
                                style={"textAlign": "center"},
                            ),
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
                            dcc.Graph(
                                figure=data1[1],
                                style={"textAlign": "center"},
                            ),
                            dcc.Graph(
                                figure=data1[2],
                                style={"textAlign": "center"},
                            ),
                            dcc.Graph(
                                figure=data1[3],
                                style={"textAlign": "center"},
                            ),
                        ],
                        width={"size": 10, "order": 1, "offset": 1},
                    ),
                ],
                align="center",
            ),

        elif active_tab == "100-Topic Dataset":
            return f"", dbc.Row(
                [
                    dbc.Col(
                        [
                            dcc.Graph(
                                figure=data1[0], style={
                                    "textAlign": "center"}
                            ),
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
                            dcc.Graph(
                                figure=data1[1], style={
                                    "textAlign": "center"}
                            ),
                            dcc.Graph(
                                figure=data1[2], style={
                                    "textAlign": "center"}
                            ),
                            dcc.Graph(
                                figure=data1[3], style={
                                    "textAlign": "center"}
                            ),
                        ],
                        width={"size": 10, "order": 1, "offset": 1},
                    ),
                ],
                align="center",
            ),

        elif active_tab == "200-Topic Dataset":
            return f"", dbc.Row(
                [
                    dbc.Col(
                        [
                            dcc.Graph(
                                figure=data1[0], style={
                                    "textAlign": "center"}
                            ),
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
                            dcc.Graph(
                                figure=data1[1], style={
                                    "textAlign": "center"}
                            ),
                            dcc.Graph(
                                figure=data1[2], style={
                                    "textAlign": "center"}
                            ),
                            dcc.Graph(
                                figure=data1[3], style={
                                    "textAlign": "center"}
                            ),
                        ],
                        width={"size": 10, "order": 1, "offset": 1},
                    ),
                ],
                align="center",
            ),

        elif active_tab == "400-Topic Dataset":
            return f"", dbc.Row(
                [
                    dbc.Col(
                        [
                            dcc.Graph(
                                figure=data1[0], style={
                                    "textAlign": "center"}
                            ),
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
                            dcc.Graph(
                                figure=data1[1], style={
                                    "textAlign": "center"}
                            ),
                            dcc.Graph(
                                figure=data1[2], style={
                                    "textAlign": "center"}
                            ),
                            dcc.Graph(
                                figure=data1[3], style={
                                    "textAlign": "center"}
                            ),
                        ],
                        width={"size": 10, "order": 1, "offset": 1},
                    ),
                ],
                align="center",
            ),

        elif active_tab == "Full Term Dataset":
            return f"", dbc.Row(
                [
                    dbc.Col(
                        [
                            dcc.Graph(
                                figure=data1[0], style={
                                    "textAlign": "center"}
                            ),
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
                            dcc.Graph(
                                figure=data1[1], style={
                                    "textAlign": "center"}
                            ),
                            dcc.Graph(
                                figure=data1[2], style={
                                    "textAlign": "center"}
                            ),
                            dcc.Graph(
                                figure=data1[3], style={
                                    "textAlign": "center"}
                            ),
                        ],
                        width={"size": 10, "order": 1, "offset": 1},
                    ),
                ],
                align="center",
            ),

    return f"", "No tab selected"


def checkif(displayoptions):
    if displayoptions == 0:
        visible = "none"
    elif displayoptions == 1:
        visible = "block"
    return visible


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
        return None, None, f"Drag and drop or select a .nii.gz or .zip archive containing fMRI data to begin"


def SaveandEncode(name, content):
    with open(join(UPLOAD_DIRECTORY, name[0]), "wb") as fp:
        fp.write(decodebytes(content[0].encode("utf8").split(b";base64,")[1]))


@app.callback(
    Output("session2", "data"),
    Output("session3", "data"),
    Output("session4", "data"),
    Output("loading-output2", "children"),
    Output("iszipped", "data"),
    Input("session", "data"),
)
def analyze(file):
    Disp = False
    if not file == None:
        if file.rsplit(".")[-1] == "zip":
            distancedict = {}
            dicts1 = {}
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
            for gs in listdir("Gradients"):

                mask, WordList, GradientList = maskingcalc(
                    WordList, GradientList, gs)
            GradientFrame = [GradientList[0], GradientList[1], GradientList[2]]
            for zzz in filelistd:
                Corrs1 = []
                WordList = []

                maskedSeries = apply_mask(
                    resample_to_img(
                        niload(zzz),
                        niload(join("Gradients", gs)),
                    ),
                    mask,
                )

                Corrs = [
                    pearsonr(GradientFrame[0], maskedSeries)[0],
                    pearsonr(GradientFrame[1], maskedSeries)[0],
                    pearsonr(GradientFrame[2], maskedSeries)[0],
                ]

                for i in listdir("CSVData"):
                    Alldistanc, top10, integer, pv_in_hex = analyzefunc(
                        Corrs, i)
                    distancedict.update({i: Alldistanc})
                print("done")
                Disp = True
                dicts1.update({zzz: Corrs})
            remove(zzz)
            iszip = True
            return distancedict, dicts1, Disp, f"File has been processed, click here to re-render the figures", iszip
        elif file.rsplit(".", 2)[0] or file.rsplit(".", 2)[1] == ".nii.gz":
            Disp = False
            ad = []

            if exists(file) == True:
                delif("assets")
                Corrs1 = []
                WordList = []
                GradientList = []
                for gs in listdir("Gradients"):

                    mask, WordList, GradientList = maskingcalc(
                        WordList, GradientList, gs
                    )
                GradientFrame = [GradientList[0],
                                 GradientList[1], GradientList[2]]

                maskedSeries = apply_mask(
                    resample_to_img(niload(file), niload(
                        join("Gradients", gs))),
                    mask,
                )

                Corrs = [
                    pearsonr(GradientFrame[0], maskedSeries)[0],
                    pearsonr(GradientFrame[1], maskedSeries)[0],
                    pearsonr(GradientFrame[2], maskedSeries)[0],
                ]
                remove(file)

                for i in listdir("CSVData"):
                    Alldistanc, top10, integer, pv_in_hex = analyzefunc(
                        Corrs, i)
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
                    wc.to_file(join("assets", "wordcloud%s.png" %
                               (i.split(".")[0])))
                    arrco = Corrs
                    Corrs1.append(list(arrco))
                    ad.append(Alldistanc)
                print("done")
                Disp = True
                distancedict = {
                    bbb: ad[em] for em, bbb in enumerate(listdir("CSVData"))
                }
                iszip = False
                return distancedict, Corrs, Disp, f"Files have been processed, click here to re-render the figures", iszip
            else:
                return no_update, no_update, Disp, no_update, no_update
        else:
            return no_update, no_update, Disp, no_update, no_update
    else:
        return no_update, no_update, Disp, no_update, no_update


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
    with open(join("CSVData", i), newline="") as f:
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
    principle_vector = array(
        df, dtype=float).reshape(-1, 1)  # turn df into array
    pv_in_hex = []
    # get the maximum absolute value in array
    vmax = abs(principle_vector).max()
    vmin = -vmax  # minimu
    # loop through each column (cap)
    for g in range(principle_vector.shape[1]):
        rescale = (principle_vector[:, g] - vmin) / \
            (vmax - vmin)  # rescale scores
        colors_hex = []
        for c in RdBu_r(rescale):
            colors_hex.append(to_hex(c))  # adds colour codes (hex) to list
        pv_in_hex.append(colors_hex)
        # add all colour codes for each item on all caps
    return Alldistanc, top10, integer, pv_in_hex


def maskingcalc(WordList, GradientList, gs):
    WordList.append(gs)
    mask = compute_background_mask(join("Gradients", gs))
    GradientList.append(apply_mask(join("Gradients", gs), mask))
    return mask, WordList, GradientList


def delif(x):
    if exists(x) == True:
        rmtree(x)
    if not (exists(x) == True):
        mkdir(x)


def GenGraphsInit(figwanted, dir="Fig_Jsons"):
    fighhh = []
    for a in figwanted:
        fig = read_json(join(dir, a))

        fig.update_layout(
            title={"text": a.split(".")[0], "xanchor": "center"},
            title_x=0.5,
            width=900,
            height=700,
            template="simple_white",
        )

        fighhh.append(fig)
        print("ooo")
    return fighhh


@app.callback(
    [
        Output("store1", "data"),
    ],
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
    colours = [
        "aliceblue",
        "aqua",
        "aquamarine",
        "azure",
        "beige",
        "bisque",
        "black",
        "blanchedalmond",
        "blue",
        "blueviolet",
        "brown",
        "burlywood",
        "cadetblue",
        "chartreuse",
        "chocolate",
        "coral",
        "cornflowerblue",
        "cornsilk",
        "crimson",
        "cyan",
        "darkblue",
        "darkcyan",
        "darkgoldenrod",
        "darkgray",
        "darkgrey",
        "darkgreen",
        "darkkhaki",
        "darkmagenta",
        "darkolivegreen",
        "darkorange",
        "darkorchid",
        "darkred",
        "darksalmon",
        "darkseagreen",
        "darkslateblue",
        "darkslategray",
        "darkslategrey",
        "darkturquoise",
        "darkviolet",
        "deeppink",
        "deepskyblue",
        "dimgray",
        "dimgrey",
        "dodgerblue",
        "firebrick",
        "floralwhite",
        "forestgreen",
        "fuchsia",
        "gainsboro",
        "ghostwhite",
        "gold",
        "goldenrod",
        "gray",
        "grey",
        "green",
        "greenyellow",
        "honeydew",
        "hotpink",
        "indianred",
        "indigo",
        "ivory",
        "khaki",
        "lavender",
        "lavenderblush",
        "lawngreen",
        "lemonchiffon",
        "lightblue",
        "lightcoral",
        "lightcyan",
        "lightgoldenrodyellow",
        "lightgray",
        "lightgrey",
        "lightgreen",
        "lightpink",
        "lightsalmon",
        "lightseagreen",
        "lightskyblue",
        "lightslategray",
        "lightslategrey",
        "lightsteelblue",
        "lightyellow",
        "lime",
        "limegreen",
        "linen",
        "magenta",
        "maroon",
        "mediumaquamarine",
        "mediumblue",
        "mediumorchid",
        "mediumpurple",
        "mediumseagreen",
        "mediumslateblue",
        "mediumspringgreen",
        "mediumturquoise",
        "mediumvioletred",
        "midnightblue",
        "mintcream",
        "mistyrose",
        "moccasin",
        "navajowhite",
        "navy",
        "oldlace",
        "olive",
        "olivedrab",
        "orange",
        "orangered",
        "orchid",
        "palegoldenrod",
        "palegreen",
        "paleturquoise",
        "palevioletred",
        "papayawhip",
        "peachpuff",
        "peru",
        "pink",
        "plum",
        "powderblue",
        "purple",
        "red",
        "rosybrown",
        "royalblue",
        "saddlebrown",
        "salmon",
        "sandybrown",
        "seagreen",
        "seashell",
        "sienna",
        "silver",
        "skyblue",
        "slateblue",
        "slategray",
        "slategrey",
        "snow",
        "springgreen",
        "steelblue",
        "tan",
        "teal",
        "thistle",
        "tomato",
        "turquoise",
        "violet",
        "wheat",
        "white",
        "whitesmoke",
        "yellow",
        "yellowgreen",
    ]
    cl1 = None
    disornot = None
    #figl = listdir("Fig_Jsons")
    figdict = {
        "Full Term Dataset": ["v5-fulldataset.json", "1,2v5-fulldataset.json", "1,3v5-fulldataset.json", "2,3v5-fulldataset.json"],
        "100-Topic Dataset": ["v5-topics-100.json", "1,2v5-topics-100.json", "1,3v5-topics-100.json", "2,3v5-topics-100.json"],
        "200-Topic Dataset": ["v5-topics-200.json", "1,2v5-topics-200.json", "1,3v5-topics-200.json", "2,3v5-topics-200.json"],
        "400-Topic Dataset": ["v5-topics-400.json", "1,2v5-topics-400.json", "1,3v5-topics-400.json", "2,3v5-topics-400.json"],
        "50-Topic Dataset": ["v5-topics-50.json", "1,2v5-topics-50.json", "1,3v5-topics-50.json", "2,3v5-topics-50.json"],
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
        if iszip == False or None:
            enu = figdict[input]
            for benu, abc in enumerate(enu):

                if abc in D3list:
                    abc = abc.split(".")[0] + ".csv"
                    ref = Scatter3d(
                        #
                        customdata=["custom point"],
                        x=[data2[0]],
                        y=[data2[1]],
                        z=[data2[2]],
                        mode="markers",
                        name="Custom",
                        marker=dict(
                            size=12,
                            color="blue",
                            symbol="diamond",
                            opacity=0.95,
                        ),
                    )
                    fig = listographs[benu]
                    fig.add_trace(ref)
                    fig.update_layout(showlegend=False)
                    figlist = [fig]
                else:
                    u = [
                        int(abc.split("v")[0].split(",")[0]) - 1,
                        int(abc.split("v")[0].split(",")[1]) - 1,
                    ]
                    ref = Scatter(
                        #
                        customdata=["custom point"],
                        x=[data2[u[0]]],
                        y=[data2[u[1]]],
                        mode="markers",
                        name="Custom",
                        marker=dict(
                            size=12,
                            color="blue",
                            symbol="diamond",
                            opacity=0.95,
                        ),
                    )
                    fig = listographs[benu]
                    fig.add_trace(ref)
                    fig.update_layout(showlegend=False)
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

                fig = listographs[benu]

                if abc in D3list:
                    Recolor(fig, graddict[input])
                    for ert, dn in enumerate(data2):
                        grpnm = dn.split("\\")[-2]
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
                            name=dn.split(".")[0].split("\\")[-1],
                            marker=dict(
                                size=12,
                                color=grpdict[grpnm],
                                symbol="diamond",
                                opacity=0.95,
                            ),
                        )

                        fig.add_trace(ref)
                        fig.update_layout(showlegend=True)

                    figlist.append(fig)
                else:
                    for est, dn in enumerate(data2):
                        grpnm = dn.split("\\")[-2]
                        est = est + ert
                        Recolor(fig, bruhdict[abc.split("v")[0]])
                        u = [
                            (int(abc.split("v")[0].split(",")[0]) - 1),
                            (int(abc.split("v")[0].split(",")[1]) - 1),
                        ]
                        ref1 = Scatter(
                            #
                            x=[data2[dn][u[0]]],
                            y=[data2[dn][u[1]]],
                            name=dn.split(".")[0].split("\\")[-1],
                            legendgroup=grpnm,
                            legendgrouptitle_text=grpnm,
                            mode="markers",
                            marker=dict(
                                size=12,
                                color=grpdict[grpnm],
                                symbol="diamond",
                                opacity=0.95,
                            ),
                        )
                        fig = listographs[benu]
                        fig.add_trace(ref1)
                        fig.update_layout(showlegend=True)
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


def Recolor(fig, cgradient):
    fig.update_traces(
        overwrite=True,
        marker_color=cgradient,
        marker_colorscale="thermal",
        marker_reversescale=True,
        selector=dict(marker_color="#636efa"),
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
    application.run(debug=True)
