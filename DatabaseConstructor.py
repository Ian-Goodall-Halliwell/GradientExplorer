
import PySimpleGUI as sg
import os
SYMBOL_UP1 =    '▲'
SYMBOL_DOWN1 =  '▼'
SYMBOL_UP2 =    '▲'
SYMBOL_DOWN2 =  '▼'
SYMBOL_UP3 =    '▲'
SYMBOL_DOWN3 =  '▼'
SYMBOL_UP4 =    '▲'
SYMBOL_DOWN4 =  '▼'
def collapse(layout, key):
    return sg.pin(sg.Column(layout, key=key, visible=False))


def DownloadandConstruct(path, filestoconstr=[False,False,False,False,False], dlneurosynth=False, dlgradients=False, dltopics=False, ):
    import os
    import nimare
    import requests
    import tarfile
    import gc
    b = 0
    print('Initializing')
    sg.OneLineProgressMeter('Initializing', b, 1000000)
    os.chdir(path)
    if (os.path.exists('tempDir') == False):
        os.mkdir('tempDir')
    if (os.path.exists('Data') == False):
        os.mkdir('Data')
    if (os.path.exists('Gradients') == False):
        os.mkdir('Gradients')
    if (os.path.exists('Topics') == False):
        os.mkdir('Topics')
    if (os.path.exists('Features') == False):
        os.mkdir('Features')
    print('Directory created')
    b = b+100
    sg.OneLineProgressMeter('Directory created',b, 1000000)
    if dlneurosynth == True:
        url1 = "https://github.com/neurosynth/neurosynth-data/blob/master/current_data.tar.gz?raw=true"
        resp1 = requests.get(url1, stream=True)
        print(resp1.headers.get('content-type'))
        o = 0
        with open('tempDir/current_data', 'wb') as fd1:
            for n1, chunk1 in enumerate(resp1.iter_content(chunk_size=128)):
                fd1.write(chunk1)
                o = o+1
                if o == 1000:
                    sg.OneLineProgressMeter('Fetching neurosynth data', b + n1, 1000000)
                    o=0
            b=b+n1
        tarfile.open('tempDir/current_data').extractall(path='Data')
        b = b+100
        sg.OneLineProgressMeter('Neurosynth data fetched', b, 1000000)
        print("Neurosynth data fetched")
        gc.collect()
        os.remove('tempDir/current_data')
    if dlgradients == True:
        url2 = "https://github.com/17iwgh/dbasepub/blob/main/GradRepo.tar?raw=true"
        resp2 = requests.get(url2, stream=True)
        with open('tempDir/Gradients', 'wb') as fd2:
            for n2, chunk2 in enumerate(resp2.iter_content(chunk_size=128)):
                fd2.write(chunk2)
                o = o+1
                if o == 10:
                    sg.OneLineProgressMeter('Fetching gradients', b+n2, 1000000)
                    o=0
            b=b+n2
        tarfile.open('tempDir/Gradients').extractall(path='Gradients')
        gc.collect()
        os.remove('tempDir/Gradients')
        b = b+100
        sg.OneLineProgressMeter('Gradients fetched', b, 1000000)
        print("Gradients fetched")
        print("Fetching Topics")
    if dltopics == True:
        url3 = "https://github.com/neurosynth/neurosynth-data/blob/master/topics/v5-topics.tar.gz?raw=true"
        resp3 = requests.get(url3, stream=True)
        with open('tempDir/Topics', 'wb') as fd3:
            for n3, chunk3 in enumerate(resp3.iter_content(chunk_size=128)):
                fd3.write(chunk3)
                o = o+1
                if o == 1000:
                    sg.OneLineProgressMeter('Fetching Topics', b+ n3, 1000000)
                    o=0
            b = b+n3
        tarfile.open('tempDir/Topics').extractall(path='Topics')
        gc.collect()
        os.remove('tempDir/Topics')
        b=b+100
        sg.OneLineProgressMeter('Topics fetched', b, 1000000)
        print("Topics fetched")
        print("Constructing topics")
        for indx, i in enumerate(os.listdir('Topics/analyses')):
            sg.OneLineProgressMeter('Constructing features', b+(indx*10000), 1000000)
            src=open(("Topics/analyses/" + i),"r") 
            fline="pm"    #Prepending string 
            oline=src.readlines() 
            #Here, we prepend the string we want to on first line 
            oline.insert(0,fline) 
            src.close() 
            #We again open the file in WRITE mode  
            src=open(("Topics/analyses/" + i),"w") 
            src.writelines(oline) 
            src.close()
            os.replace(("Topics/analyses/" + i), ("Features/" + i)) 
            #We read the existing text from file in READ mode 
        b=b+(indx*10000)
        os.replace("Data/features.txt", "Features/v5-fulldataset.txt")
        b=b+100
        sg.OneLineProgressMeter('Features constructed', b, 1000000)
        print("Features constructed")

    if not filestoconstr == [False,False,False,False,False]:
        if (os.path.exists('Packaged_Datasets') == False):
            os.mkdir('Packaged_Datasets')

        print("Creating packaged datasets. Go get a coffee and put on a movie, this can take an hour or more...")
        for idx, y in enumerate(os.listdir('Features')):
            if filestoconstr[idx] == True:
                sg.OneLineProgressMeter('Creating packaged datasets. Go get a coffee and put on a movie, this can take an hour or more...', b+(idx*100000), 1000000)
                ns_dset = nimare.io.convert_neurosynth_to_dataset("Data/database.txt",annotations_file=("Features/" + y))
                ns_dset.save("Packaged_Datasets/%s.pkl" % (y.split(".")[0])) 
        sg.OneLineProgressMeter('Done!', 1000000, 1000000)
        print("Done!")

def CSVComp(path, csvstoconstr=[False,False,False,False,False]):
    import os
    if not csvstoconstr == [False,False,False,False,False]:
        os.chdir(path)
        import gc
        from nilearn.image import resample_to_img
        import pandas as pd
        import os
        import nimare
            
        if (os.path.exists('CSVLoadings') == False):
            os.mkdir('CSVLoadings')

        u=0
        for idx, ds in enumerate(os.listdir('Packaged_Datasets')):
            if csvstoconstr[idx] == True:
                ns_dset = nimare.dataset.Dataset.load(os.path.join("Packaged_Datasets", ds))
                terms = ns_dset.get_labels()
                CutList = pd.DataFrame()
                for tidx, i in enumerate(terms):
                    Corrs = []
                    term_dset = ns_dset.slice(ns_dset.get_studies_by_label(i, label_threshold=0.00000001))
                    termframe = term_dset.get({'annotations': ('annotations', None)})
                    TermAnnots = termframe['annotations']
                    Tlist = []
                    for dd in TermAnnots:
                        dg = dd[i].to_numpy()
                        dst = dd['id']
                        dstr = str(dst).split("    ")[1].split("\n")[0]
                        dt = [dstr, str(dg)]
                        Tlist.append(dt) 
                    CutList = pd.DataFrame(data=Tlist, columns=[('id_'+ i), i])
                    Sortlist = CutList
                    
                    sg.OneLineProgressMeter('Working...',u+tidx, 3700)
                    if tidx == 0:
                        Sortlist.to_csv(os.path.join('CSVLoadings', 'Loadingsfor%s.csv' % (ds).split(".")[0]), header=Sortlist.columns)
                    else:
                        df = pd.read_csv(os.path.join('CSVLoadings', 'Loadingsfor%s.csv' % (ds).split(".")[0]), dtype=str)
                        new_column = Sortlist
                        df1 = pd.concat((df, new_column), axis=1)
                        df1.to_csv(os.path.join('CSVLoadings', 'Loadingsfor%s.csv' % (ds.split(".")[0])), index = False, header=df1.columns)
                    Sortlist = []
                    print("done ", i)
                    gc.collect()

def Construct(path, dsetstocontr=[False,False,False,False,False]):
    from operator import index
    import shutil
    from pandas.io.pytables import Term
    import nimare
    import os
    import numpy as np
    from nilearn import masking
    import gc
    import scipy
    import csv
    import nibabel as nib
    from nilearn.image import resample_to_img
    import pandas as pd
    import zipfile
    os.chdir(path)
    if (os.path.exists('TermMaps') == True):
        shutil.rmtree('TermMaps')
    if (os.path.exists('CSVData') == False):
        os.mkdir('CSVData')
    if (os.path.exists('TermMaps') == False):
        os.mkdir('TermMaps')
    if (os.path.exists('TermSets') == False):
        os.mkdir('TermSets')
    if (os.path.exists('Backup_Data') == False):
        os.mkdir('Backup_Data')
    meta = meta = nimare.meta.cbma.mkda.MKDADensity()
    GradientList = []
    WordList = []
    maptype = 'stat'

    for idx, gs in enumerate(os.listdir('Gradients')):
        wordir = os.path.join("Gradients", gs)
        WordList.append(gs)
        if (idx == 0):
            mask = masking.compute_background_mask(wordir) 
        MaskedGradient = masking.apply_mask(wordir,mask)
        GradientList.append(MaskedGradient)
    GradientFrame = pd.DataFrame(GradientList, index=WordList, dtype='float16').T
    SampleGradient = nib.load(wordir)
    FullList = []
    TermList = []
    u=0
    for idx, ds in enumerate(os.listdir('Packaged_Datasets')):
        if dsetstocontr[idx] == True:
            ns_dset = nimare.dataset.Dataset.load(os.path.join("Packaged_Datasets", ds))
            terms = ns_dset.get_labels()
            TermList = []
            FullList = []
            shutil.rmtree('TermMaps')
            os.mkdir('TermMaps')
            for tidx, i in enumerate(terms):
                curset = (pd.read_csv(os.path.join('CSVLoadings', ("Loadingsfor" + ds.split(".")[0]+".csv")), dtype='str', usecols=[("id_"+i), i])[[("id_"+i), i]]).dropna()
                if len(curset) >= 999:
                    cursort = curset.sort_values(by=i, ascending=False)
                    cursort.reset_index(inplace=True, drop=True)
                    cursort = cursort.truncate(after=999)
                    curset = cursort
                    print(i)
                term_dset = ns_dset.slice(curset[("id_"+i)])
                Corrs = []
                results = meta.fit(term_dset)
                results.save_maps(output_dir='TermMaps', prefix=i, prefix_sep='__')
                maskedResult = masking.apply_mask(resample_to_img(results.get_map(maptype), SampleGradient), mask)
                maskedSeries = pd.Series(maskedResult, dtype='float16')
                #maskedArray = np.asarray(maskedResult, dtype='float16')
                Corrs = GradientFrame.corrwith(maskedSeries).to_list()
                FullList.append(Corrs)
                TermList.append(i.split("__")[1])
                
                sg.OneLineProgressMeter('Working...',u+tidx, 3700)
                print("Completed ", i, ", correlation number ", tidx, "of ", len(terms), "in ", ds)
                gc.collect()
            u = u + tidx
            with open(os.path.join('CSVData', '%s.csv'  % (ds.split(".")[0])), 'w') as f:
                writer = csv.writer(f)
                writer.writerows(zip(TermList, FullList))
            with zipfile.ZipFile(os.path.join('Backup_Data', '%s.zip' % (ds.split(".")[0])), 'w') as zipF:
                for file in os.listdir('TermMaps'):
                    zipF.write(os.path.join("TermMaps/" + file), compress_type=zipfile.ZIP_DEFLATED)
            gc.collect()
            print(ds, " complete! Starting next one now...")

section1 = [[sg.Text('Select datasets to fetch and prepare',text_color='black')],
            [sg.Checkbox('Neurosynth term database',text_color='black',default=False, enable_events=True, key='-term'),sg.Checkbox('Default gradient database', text_color='black', default=False, enable_events=True, key='-grad'),sg.Checkbox('Neurosynth topic database',text_color='black',default=False, enable_events=True, key='-top')],
            [sg.Checkbox('50-topic subset',text_color='black',default=False, enable_events=True, key='-c1'),sg.Checkbox('100-topic subset',text_color='black',default=False,enable_events=True, key='-c2'),sg.Checkbox('200-topic subset', text_color='black',default=False,enable_events=True, key='-c3'),sg.Checkbox('400-topic subset',text_color='black',default=False, enable_events=True, key='-c4'),sg.Checkbox('Full neurosynth term database (WILL TURN COMPUTER INTO A BRICK FOR ~6 HOURS)',text_color='black',default=False, enable_events=True, key='-c5')]]

section2 = [[sg.Text('Select datasets to preprocesss for analysis',text_color='black')],
            [sg.Checkbox('50-topic database',text_color='black',default=False, enable_events=True, key='-d1'),sg.Checkbox('100-topic database',text_color='black',default=False,enable_events=True, key='-d2'),sg.Checkbox('200-topic database',text_color='black', default=False,enable_events=True, key='-d3'),sg.Checkbox('400-topic database',text_color='black',default=False, enable_events=True, key='-d4'),sg.Checkbox('Full neurosynth term database',text_color='black',default=False, enable_events=True, key='-d5')]]

section3 = [[sg.Text('Select datasets to analyze',text_color='black')],
            [sg.Checkbox('50-topic dataset',text_color='black',default=False, enable_events=True, key='-e1'),sg.Checkbox('100-topic dataset',text_color='black',default=False,enable_events=True, key='-e2'),sg.Checkbox('200-topic dataset', text_color='black',default=False,enable_events=True, key='-e3'),sg.Checkbox('400-topic dataset',text_color='black',default=False, enable_events=True, key='-e4'),sg.Checkbox('Full neurosynth term dataset',text_color='black',default=False, enable_events=True, key='-e5')]]

section4 = [[sg.Text(SYMBOL_UP1, enable_events=True, k='-OPEN SEC1-', text_color='black'),
             sg.Text('Database fetching', enable_events=True, text_color='black', k='-OPEN SEC1-TEXT')],
            [collapse(section1, '-SEC1-')],
            [sg.Text(SYMBOL_UP2, enable_events=True, k='-OPEN SEC2-', text_color='black'),
             sg.Text('Database preprocessing', enable_events=True, text_color='black', k='-OPEN SEC2-TEXT')],
            [collapse(section2, '-SEC2-')],
            [sg.Text(SYMBOL_UP3, enable_events=True, k='-OPEN SEC3-', text_color='black'),
             sg.Text('Analysis', enable_events=True, text_color='black', k='-OPEN SEC3-TEXT')],
            [collapse(section3, '-SEC3-')],
            [sg.Button('Go'), sg.Button('Exit')],
            [sg.Output(size=(60,10))]]

layout = [[sg.Text('Choose directory to save files (ideally an empty one)', text_color='black')],
          [sg.FolderBrowse(enable_events=True, key='--path', k='-OPEN SEC4-TEXT', initial_folder=os.getcwd())],
          [sg.Text(SYMBOL_UP4, enable_events=True, k='-OPEN SEC4-', text_color='black'),
           sg.Text('Data construction options', enable_events=True, text_color='black', k='-OPEN SEC4-TEXT')],
          [collapse(section4, '-SEC4-')]]

window = sg.Window('Computer-melter 9000', layout)
opened1, opened2, opened3, opened4 = False, False, False, False
while True:             # Event Loop
    event, values = window.read()
    
    if event == sg.WIN_CLOSED or event == 'Exit':
        break
    if event.startswith('-OPEN SEC4-'):
        opened4 = not opened4
        window['-OPEN SEC4-'].update(SYMBOL_DOWN4 if opened4 else SYMBOL_UP4)
        window['-SEC4-'].update(visible=opened4)
    if event.startswith('-OPEN SEC1-'):
        opened1 = not opened1
        window['-OPEN SEC1-'].update(SYMBOL_DOWN1 if opened1 else SYMBOL_UP1)
        window['-SEC1-'].update(visible=opened1)
    if event.startswith('-OPEN SEC2-'):
        opened2 = not opened2
        window['-OPEN SEC2-'].update(SYMBOL_DOWN2 if opened2 else SYMBOL_UP2)
        window['-SEC2-'].update(visible=opened2)
    if event.startswith('-OPEN SEC3-'):
        opened3 = not opened3
        window['-OPEN SEC3-'].update(SYMBOL_DOWN3 if opened3 else SYMBOL_UP3)
        window['-SEC3-'].update(visible=opened3)
    if event == 'Go':
        if values["--path"] == 0:
            print("Please select a directory")
            break
        if not [values["-c5"],values["-c1"],values["-c2"],values["-c3"],values["-c4"]] == [False,False,False,False,False]:
            DownloadandConstruct(values["--path"], dlneurosynth=values["-term"],dlgradients=values["-grad"],dltopics=values["-top"],filestoconstr=[values["-c5"],values["-c1"],values["-c2"],values["-c3"],values["-c4"]])
            print('Done downloads/construction')
        if not [values["-d5"],values["-d1"],values["-d2"],values["-d3"],values["-d4"]] == [False,False,False,False,False]:
            CSVComp(values["--path"], csvstoconstr=[values["-d5"],values["-d1"],values["-d2"],values["-d3"],values["-d4"]])
            print('Done pre-proccessing!')
        if not [values["-e5"],values["-e1"],values["-e2"],values["-e3"],values["-e4"]] == [False,False,False,False,False]:
            Construct(values["--path"], dsetstocontr=[values["-e5"],values["-e1"],values["-e2"],values["-e3"],values["-e4"]])
            print('Done analysis!')
    else:
        print(event, values)
window.close()