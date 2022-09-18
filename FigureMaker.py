import pandas as pd
import csv
import os
import plotly.graph_objects as go

path1 = os.path.dirname(os.path.abspath(__file__)) + "//Empty_Fig_Jsons"
sources = [
    ["v5-topics-50.csv", "v5-topics-50.json"],
    ["v5-topics-100.csv", "v5-topics-100.json"],
    ["v5-topics-200.csv", "v5-topics-200.json"],
    ["v5-topics-400.csv", "v5-topics-400.json"],
    ["v5-fulldataset.csv", "v5-fulldataset.json"],
]

empty = True


def savefig(path, source, source2, empty):
    import pandas as pd
    import os
    import plotly_express as px
    import plotly.io as io
    import json

    if not os.path.exists(path) == True:
        os.mkdir(path)
    data = pd.read_csv(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)) + "//CSVData//", source
        ),
        header=None,
    )
    datlis = []
    for ind, i in data.iterrows():
        rowval = i[1].strip("][").split(", ")
        for id, d in enumerate(rowval):
            rowval[id] = float(rowval[id])
        datlis.append(rowval)
        rowval = []
    cols = ["Gradient 1", "Gradient 2", "Gradient 3"]
    if empty == True:
        datlis = []

    df = pd.DataFrame(data=datlis, index=None, columns=cols, dtype=None, copy=None)
    if empty == False:
        df["Termnames"] = data[0]
    else:
        df["Termnames"] = []
    fig = px.scatter_3d(
        df, x="Gradient 1", y="Gradient 2", z="Gradient 3", hover_name="Termnames"
    )

    io.write_json(fig, os.path.join(path1, source2))
    fig.show()


def savefig2d(path, source, source2, empty):
    import pandas as pd
    import os
    import plotly_express as px
    import plotly.io as io
    import json

    if not os.path.exists(path) == True:
        os.mkdir(path)
    data = pd.read_csv(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)) + "//CSVData//", source
        ),
        header=None,
    )
    datlis = []
    for ind, i in data.iterrows():
        rowval = i[1].strip("][").split(", ")
        for id, d in enumerate(rowval):
            rowval[id] = float(rowval[id])
        datlis.append(rowval)
        rowval = []
    cols = ["Gradient 1", "Gradient 2", "Gradient 3"]
    if empty == True:
        datlis = []
    df = pd.DataFrame(data=datlis, index=None, columns=cols, dtype=None, copy=None)
    if empty == False:
        df["Termnames"] = data[0]
    else:
        df["Termnames"] = []
    fig12 = px.scatter(df, x="Gradient 1", y="Gradient 2", hover_name="Termnames")
    fig13 = px.scatter(df, x="Gradient 1", y="Gradient 3", hover_name="Termnames")
    fig23 = px.scatter(df, x="Gradient 2", y="Gradient 3", hover_name="Termnames")
    sourcefig = "1,2" + source2
    io.write_json(fig12, os.path.join(path1, sourcefig))
    sourcefig = "1,3" + source2
    io.write_json(fig13, os.path.join(path1, sourcefig))
    sourcefig = "2,3" + source2
    io.write_json(fig23, os.path.join(path1, sourcefig))


def fixer():

    for en, i in enumerate(os.listdir("CSVData")):
        Dframe = []
        if not en == 0:
            with open(os.path.join("CSVData", i), newline="") as f:
                reader = csv.reader(f)
                Dlist = list(reader)
            for rowl in Dlist:
                if not rowl == []:
                    rowl[0] = rowl[0].split("_", 1)[1]
                    Dframe.append(rowl)
        dff = pd.DataFrame(Dframe)
        dff.to_csv(i + "new" + ".csv")


for source in sources:
    source1 = source[0]
    source2 = source[1]
    savefig(path1, source1, source2, empty)

    savefig2d(path1, source1, source2, empty)


def openfig(path):
    import pandas as pd
    import json
    import plotly_express as px
    import plotly.io as io

    with open(path) as f:
        jsondata = json.load(f)
    fig = io.from_json(jsondata)
    # fig.show()


# openfig(os.path.join(path1,source2))
