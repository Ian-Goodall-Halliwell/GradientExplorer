import csv
import wordclouds_for_caps_neurosynth
from matplotlib.pyplot import title
with open("C:/Users/Ian/Documents/GitHub/Servertest/Cap analysis/pca_new.csv") as file:
    reader = csv.reader(file)
    pcadict = {}
    for en,row in enumerate(reader):
        if en == 0:
            del row[0]
            titles = row
            continue
        pcat = row[0]
        del row[0]
        for e,item in enumerate(row):
            row[e] = float(item)
        pcadict.update({pcat:row})
for e,value in enumerate(pcadict):
    if e == 3:
        for en,a in enumerate(pcadict[value]):
            pcadict[value][en] = -1*a
    wordclouds_for_caps_neurosynth.wordclouder(pcadict[value], titles,value)
    print('done')