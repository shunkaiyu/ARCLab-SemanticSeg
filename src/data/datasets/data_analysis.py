import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2,os
import argparse

parser = argparse.ArgumentParser(description='data analysis and visualization')
parser.add_argument('--folder1', default='temp', type=str)
parser.add_argument('--folder2', default='temp', type=str)
parser.add_argument('--folder3', default='temp', type=str)
parser.add_argument('--save_dir', default='temp', type=str)

def main():    
    global args
    args, unknown = parser.parse_known_args()
    print(args)

    path1 = args.folder1    
    path2 = args.folder2
    path3 = args.folder3

    #Read classes category and statistics files  
    os.chdir("/home/arcseg/Desktop/Shunkai-working/src/data/datasets")
    f = open('cholecSegClasses.json')
    data = json.load(f)
    classes = [data['classes'][x]['name'] for x in range(0,13)]
    file_name = 'cholecSeg8k.xlsx' 
    df = pd.read_excel(file_name)

    #Create df of 3 folders
    videos = df.columns
    videos = videos[2:-1]
    j = videos[3][-2:]

    df['part1'] = 0
    for v in videos: 
        i = v[-2:]
        if int(i) in [1,35,43,25,20,9]:
            df['part1']+=df['video'+i]
    df['part2'] = 0
    for v in videos: 
        i = v[-2:]
        if int(i) in [12,37,48,27,28]:
            df['part2']+=df['video'+i]
    df['part3'] = 0
    for v in videos: 
        i = v[-2:]
        if int(i) in [17,55,26,18,52,24]:
            df['part3']+=df['video'+i]

    df = df.rename(columns={"part1": "fold 1", "part2": "fold 2", "part3": "fold 3", "invalid": "Invalid"}) 

    df['fold 1&2'] = df['fold 1']+df['fold 2']
    df['fold 1&3'] = df['fold 1']+df['fold 3']
    df['fold 2&3'] = df['fold 2']+df['fold 3']
    df['entire dataset'] = df['fold 1']+df['fold 2']+df['fold 3']

    brokenPlot('fold 1','Label',df,1000000,5000000,1000000000, "Fold 1 Class Distribution")
    brokenPlot('fold 2','Label',df,1000000,5000000,1000000000, "Fold 2 Class Distribution")
    brokenPlot('fold 3','Label',df,1000000,5000000,1000000000, "Fold 3 Class Distribution")
    brokenPlot('entire dataset','Label',df,5000000,1000000,1000000000, "Entire Dataset Class Distribution")

    #Get name id to number of pixel map
    color2id = dict()
    color2num = dict()
    id2num = dict()
    classes = [data['classes'][x]['name'] for x in range(0,13)]
    colors = [data['classes'][x]['color'] for x in range(0,13)]
    for i in range(0,13):
        color2id[colors[i]] = classes[i]
    for i in range(0,13):
        color2num[colors[i]] = 0
        id2num[classes[i]] = 0
    id2num['invalid'] = 0

    #Read groundtruth path
    #test1 = './'+path+'/test/groundtruth'
    cholec_p1 = './'+path1+'/test/groundtruth'
    cholec_p2 = './'+path2+'/test/groundtruth'
    cholec_p3 = './'+path3+'/test/groundtruth'
    cholec_list = [cholec_p1,cholec_p2,cholec_p3]
    id2num_1 = id2num.copy()
    id2num_2 = id2num.copy()
    id2num_3 = id2num.copy()

    #Get the num of labels
    id2num_1 = getSetLabel(cholec_p1,id2num_1,color2id)
    id2num_2 = getSetLabel(cholec_p2,id2num_2,color2id)
    id2num_3 = getSetLabel(cholec_p3,id2num_3,color2id)

    #To dataframe
    data1 = pd.DataFrame.from_dict(id2num_1,orient='index')
    data2 = pd.DataFrame.from_dict(id2num_2,orient='index')
    data3 = pd.DataFrame.from_dict(id2num_3,orient='index')

    #Calculate frequency
    newdata = pd.concat([data1, data2], axis=1)
    newdata = pd.concat([newdata, data3], axis=1)
    newdata.columns = ['fold 3','fold 2','fold 1']
    newdata = newdata.rename(columns={"invalid": "Invalid"}) 
    classes = [data['classes'][x]['name'] for x in range(0,13)]
    newdata['Label'] = ""
    for i in range(0,13):
        newdata['Label'][i] = classes[i]
    newdata['Label'][13] = "Invalid"
    newdata['total'] = newdata['fold 1']+newdata['fold 3']+newdata['fold 2']
    newdata['fold 1&2'] = (newdata['fold 1']+newdata['fold 2'])#/5600
    newdata['fold 1&3'] = (newdata['fold 1']+newdata['fold 3'])#/5600
    newdata['fold 2&3'] = (newdata['fold 3']+newdata['fold 2'])#/(2480+2800)
    newdata['entire dataset'] = newdata['total']#/(5600+2480)
    #newdata = newdata.rename(columns={"part1": "fold 1", "part2": "fold 2", "part3": "fold 3"})
    newdata['fold 1'] = (newdata['fold 1'])#/2800
    newdata['fold 2'] = (newdata['fold 2'])#/2800
    newdata['fold 3'] = (newdata['fold 3'])#/(2480)

    #Add Class Column
    classes = [data['classes'][x]['name'] for x in range(0,13)]
    for i in range(0,13):
        newdata['Label'][i] = classes[i]
    newdata['Label'][13] = "Invalid"

    brokenPlot('fold 1','Label',newdata,1000,1500,3000, "Fold 1 Class Frequency")
    brokenPlot('fold 2','Label',newdata,1000,1500,3000, "Fold 2 Class Frequency")
    brokenPlot('fold 3','Label',newdata,1000,1500,3000, "Fold 3 Class Frequency")
    brokenPlot('entire dataset','Label',newdata,500,1000,10000, "Entire Dataset Class Frequency")

#Broken Plot
def brokenPlot(dx,dy,df,leftHighLim,lowLim,highLim,title):
    new_style = {'grid': False}
    sns.set_style("white")
    f, (ax,ax2) = plt.subplots(ncols=2, nrows=1, sharex='col',figsize=(15,10))

    # plot the same data on both axes
    ax = sns.barplot(x=dx, y=dy, data=df, ax=ax, orient='h')
    ax2 = sns.barplot(x=dx, y=dy, data=df, ax=ax2, orient='h')

    # zoom-in / limit the view to different portions of the data
    ax2.set_xlim(lowLim, highLim)  # outliers only
    ax.set_xlim(0, leftHighLim)  # most of the data

    # hide the spines between ax and ax2
    ax.spines['left'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax.yaxis.tick_left()
    ax2.get_yaxis().set_visible(False)
    ax2.set_xlabel('Num of Pixels')
    #ax2.tick_params(labelleft=False)  # don't put tick labels at the top
    #ax2.yaxis.tick_right()
    ax.set_ylabel("Classes")
    ax2.set_ylabel("")

    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((1-d, 1+d), (1-d, 1+d), **kwargs)       
    ax.plot((1-d, 1+d), (-d, +d), **kwargs) 

    kwargs.update(transform=ax2.transAxes)  
    ax2.plot((-d, +d), (1- d, 1+ d), **kwargs)  
    ax2.plot((-d, +d), (-d, +d), **kwargs)  
    
    ax.invert_yaxis()
    ax2.invert_yaxis()
    
    plt.title = title
    #plt.show()
    plt.savefig(args.save_dir+title+".png")

def getLabel(img, color2id):
    label = set()
    h,w,c = img.shape
    for i,j in zip(range(0,h),range(0,w)):
        pixel = img[i,j,:]
        pixel = pixel[::-1]
        #print(pixel.shape)
        color = ','.join(str(e) for e in pixel)
        color = '['+color+']'
        #label.add(color)
        if color in color2id.keys():
            #print(color)
            label.add(color2id[color])
        else:
            label.add('invalid')
    return label

def getSetLabel(path, id2num,color2id):
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path,filename))
        labels = getLabel(img,color2id)
        for l in labels:
            id2num[l] += 1
    return id2num

if __name__ == '__main__':
    main()
