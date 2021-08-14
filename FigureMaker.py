import os
import plotly.graph_objects as go

path1 = "Fig_Jsons"
source1 = "v5-fulldataset.csv"
source2 = "v5-fulldataset.json"


def savefig(path, source):
    import pandas as pd
    import os
    import plotly_express as px
    import plotly.io as io
    import json

    if not os.path.exists(path) == True:
        os.mkdir(path)
    data = pd.read_csv(os.path.join("CSVData/", source), header=None)
    datlis = []
    for ind, i in data.iterrows():
        rowval = i[1].strip("][").split(", ")
        for id, d in enumerate(rowval):
            rowval[id] = float(rowval[id])
        datlis.append(rowval)
        rowval = []
    cols = ["Gradient 1", "Gradient 2", "Gradient 3"]
    df = pd.DataFrame(data=datlis, index=None, columns=cols, dtype=None, copy=None)
    df["Termnames"] = data[0]
    fig = px.scatter_3d(
        df, x="Gradient 1", y="Gradient 2", z="Gradient 3", hover_name="Termnames"
    )
    # jsonfig = io.to_json(fig)
    # with open(os.path.join(path1,source2), 'w') as f:
    # json.dumps(jsonfig, f)
    # with open(path) as f:
    # jsondata = json.dumps(f)
    io.write_json(fig, os.path.join(path1, source2))
    fig.show()


def savefig2d(path, source):
    import pandas as pd
    import os
    import plotly_express as px
    import plotly.io as io
    import json

    if not os.path.exists(path) == True:
        os.mkdir(path)
    data = pd.read_csv(os.path.join("CSVData/", source), header=None)
    datlis = []
    for ind, i in data.iterrows():
        rowval = i[1].strip("][").split(", ")
        for id, d in enumerate(rowval):
            rowval[id] = float(rowval[id])
        datlis.append(rowval)
        rowval = []
    cols = ["Gradient 1", "Gradient 2", "Gradient 3"]
    df = pd.DataFrame(data=datlis, index=None, columns=cols, dtype=None, copy=None)
    df["Termnames"] = data[0]
    fig12 = px.scatter(df, x="Gradient 1", y="Gradient 2", hover_name="Termnames")
    fig13 = px.scatter(df, x="Gradient 1", y="Gradient 3", hover_name="Termnames")
    fig23 = px.scatter(df, x="Gradient 2", y="Gradient 3", hover_name="Termnames")
    sourcefig = "1,2" + source2
    io.write_json(fig12, os.path.join(path1, sourcefig))
    sourcefig = "1,3" + source2
    io.write_json(fig13, os.path.join(path1, sourcefig))
    sourcefig = "2,3" + source2
    io.write_json(fig23, os.path.join(path1, sourcefig))
    # jsonfig = io.to_json(fig)
    # with open(os.path.join(path1,source2), 'w') as f:
    # json.dumps(jsonfig, f)
    # with open(path) as f:
    # jsondata = json.dumps(f)


savefig(path1, source1)

# savefig2d(path1, source1)


def openfig(path):
    import pandas as pd
    import json
    import plotly_express as px
    import plotly.io as io

    with open(path) as f:
        jsondata = json.load(f)
    fig = io.from_json(jsondata)
    fig.show()


# openfig(os.path.join(path1,source2))
