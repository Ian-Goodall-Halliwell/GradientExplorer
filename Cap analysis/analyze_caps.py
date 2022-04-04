import csv
import os
import numpy as np
import pandas as pd
import plotly.express as px
import wordclouds_for_caps_neurosynth
from collections import OrderedDict
def distance(p1,p2):
    squared_dist = np.sum((p1-p2)**2, axis=0)
    return np.sqrt(squared_dist)
caplist = []
tasklist = []
with open("C:/Users/Ian/Documents/GitHub/Servertest/Cap analysis/Coords.csv") as file:
    reader = csv.reader(file)
    for line in reader:
        if line != []:
            if line[0] == 'Map Name':
                continue
            if line[0].split('_')[0] =='cap':
                caplist.append(line)
            else:
                tasklist.append(line)
for enr,cap in enumerate(caplist):
    capr = np.array([float(cap[1]),float(cap[2]),float(cap[3])])
    caplis = []
    lablis = []
    for task in tasklist:
        taskr = np.array([float(task[1]),float(task[2]),float(task[3])])
        caplis.append(1/distance(capr,taskr))
        lablis.append(task[0])
        
    df = pd.DataFrame(
        dict(
            caps = caplis,
            labels = lablis
        )
    )
    cpd = OrderedDict()
    cpd['neurosynth'] = caplis
    wordclouds_for_caps_neurosynth.wordclouder(caplis,lablis,enr,savefile=True)
    fig = px.line_polar(df, r='caps', theta='labels', line_close=True,title=cap[0],range_r=[0,3])
    fig.update_traces(fill='toself')
    #fig.show()
    fig.write_image("C:/Users/Ian/Documents/GitHub/Servertest/Cap analysis/{}.jpg".format(cap[0]))
    print('stop')