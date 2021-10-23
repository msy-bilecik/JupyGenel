import json
from glob import glob
import cv2
import skimage, os
import numpy as np # linear algebra
import pandas as pd
from skimage import img_as_ubyte
import Metric_eval as metreE


def img2jsonAnnEx(sourcePath,imgPatch,maskSource,maskPatch,maskLabel,showlist=True,fileName="via_region_data.json"):
    all_images = sorted(glob(os.path.join(sourcePath,'*'+imgPatch+'*')))
    all_masks  = sorted(glob(os.path.join(maskSource,'*'+maskPatch+'*')))
    jsonF      = images2json(all_images,all_masks,maskLabel,showlist)
    file       = open(sourcePath+fileName, 'w') 
    file.write(jsonF) 
    file.close() 
    print(sourcePath+fileName+" json ok")

    return True


def img2jsonAnnPre(sourcePath,imgPatch,maskSource,maskPatch,maskLabel,showlist=True,fileName="via_region_data.json"):
    all_images =sorted(glob(os.path.join(sourcePath,imgPatch+'*')))
    all_masks  =sorted(glob(os.path.join(maskSource,maskPatch+'*')))
    jsonF=images2json(all_images,all_masks,maskLabel,showlist)
    file = open(sourcePath+fileName, 'w') 
    file.write(jsonF) 
    file.close() 
    print(sourcePath+fileName+" json ok")

    return True



def images2json(all_images,all_masks,maskLabel,showlist=True):
    print(len(all_masks))
    for i in range(len(all_images)):
        fname=os.path.basename(all_images[i])
        mname=os.path.basename(all_masks[i])
        if (showlist):
            print("{} - {} / {}".format(i,fname,mname))
    data = {}
    i=0
    for item in all_masks:
        img=all_images[i]
        sizeX=os.stat(img).st_size
        fname=os.path.basename(img)
        fnameA=fname.split("_")
        key=str(fname)+""+str(sizeX)
        value={}
        value["fileref"]=""
        value["size"]=sizeX
        value["filename"]=fname
        value["base64_img_data"]=""
        fileattributes={}
        value["file_attributes"]=fileattributes

        regions={}
        shape_attributes={}
        region_attributes={}
        region_attributes["label"]=maskLabel
        
        src = cv2.imread(item)
        #print(np.histogram(src))

        gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
        # print(np.histogram(gray))
        blur = cv2.blur(gray, (10, 10))
        ret, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        cnt = sorted(contours, key=cv2.contourArea, reverse=False)
                # ROI will be object with biggest contour
        #mask = contours[0]
        #print(len(cnt))
        c1=0
        for item in cnt:
            xs=[]
            ys=[]
            cc=0
            all_points_x={}
            all_points_y={}
            for ix in item:
                xs.append(ix[0][0])
                ys.append(ix[0][1])
                all_points_x[str(cc)]=ix[0][0]
                all_points_y[str(cc)]=ix[0][1]
                cc=cc+1
            dd={}
            shape_attributes={}
            shape_attributes["name"]="polygon"
            xs.append(xs[0])
            ys.append(ys[0])
            shape_attributes["all_points_x"]=str(xs)
            shape_attributes["all_points_y"]=str(ys)
            dd["shape_attributes"]=shape_attributes
            dd["region_attributes"]=region_attributes
            
            regions[str(c1)]=dd
            c1=c1+1
            
        value["regions"]=regions
        data[key]=value    
        i=i+1
        
    json_data = json.dumps(data)
    json_data=json_data.replace('"[','[')
    json_data=json_data.replace(']"',']')
    json_data=json_data.replace('}}}}, "','}}}}, \n "')    
    return json_data

def data2csv(fileData, path,fname="summary.csv",sep=";"):
    df = pd.DataFrame(fileData) 
    fName=os.path.join(path,fname)
    print(fName+" file saved.") 
    df.to_csv(fName,sep)

    return True

def matrixPredict(npyfile,flag_multi_class = False,num_class = 2):
    imgs=[]
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        img = img_as_ubyte(img)
        imgs.append(img)
    return imgs     
    

def matrixMask(augMaskTestDir,maskPatch="mask"):
    augMaskTestDirImages =sorted(glob(os.path.join(augMaskTestDir,'*'+maskPatch+'*')))
    mask_arr = []
    maskNames_arr= []
    for i in range(len(augMaskTestDirImages)):
        img = cv2.imread(augMaskTestDirImages[i])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = img_as_ubyte(img)
        mask_arr.append(img)
        maskNames_arr.append(os.path.basename(augMaskTestDirImages[i]).split('.')[0])
    return mask_arr, maskNames_arr



def UnetTestW(augImageTestDirImages,GTmasks,predicts,csvPath,csvFile="abc.csv",kontrol=1):
    DSCx  = []
    JACx  = []
    IOUx  = []
    VOEx  = []
    ASDx  = []
    ASSDx = []
    RAVDx = []
    df    = pd.DataFrame()
    for i in range(len(augImageTestDirImages)):
        info       = os.path.basename(augImageTestDirImages[i])
        result     = predicts[i]
        reference  = GTmasks[i]
        testX      = metricUnet(result,reference)
        
        allXX={'image_id':i,'info_id': info}
        allXX.update(testX)
        dfA=pd.DataFrame(allXX,index=[0])
        df=pd.concat([df,dfA],ignore_index=True)    

        if(kontrol ==1 ):
            print(dfA)
        DSCx.append(testX["dc"])
        JACx.append(testX["jc"])
        IOUx.append(testX["iou"])
        VOEx.append(testX["voe"])
        ASDx.append(testX["asd"])
        ASSDx.append(testX["assd"])
        RAVDx.append(testX["ravd"])
    DSC  = np.average(DSCx)
    JAC  = np.average(JACx)
    IOU  = np.average(IOUx)
    VOE  = np.average(VOEx)
    ASD  = np.average(ASDx)
    ASSD = np.average(ASSDx)
    RAVD = np.average(RAVDx)
    
    if not csvFile=="abc.csv":
        data2csv(df, csvPath,fname=csvFile,sep=";")
    
    return {"DSC"  :DSC,
            "JAC"  :JAC,
            "IOU"  :IOU,
            "VOE"  :VOE,
            "ASD"  :ASD,
            "ASSD" :ASSD,
            "RAVD" :RAVD,
           }

def metricUnet(result,reference):
    dc  = metreE.dc(result, reference)
    iou = cal_iou(result,reference)
    jc  = metreE.jc(result, reference)
    voe = 1-iou
    vol = np.count_nonzero(result)
    if not (vol==0):
        asd = metreE.asd(result,reference)
        assd= metreE.assd(result,reference)
    else:
        asd = metreE.obj_asd(result,reference)
        assd= metreE.obj_assd(result,reference)
    ravd= metreE.ravd(result,reference)
    
    return { "dc"  : dc,
            "jc"   : jc,    
            "iou"  : iou,
            "voe"  : voe,
            "asd"  : asd,
            "assd" : assd,
            "ravd" : ravd
           }

def cal_iou(result,reference):
    intersection =cv2.bitwise_and(result,reference)
    union           =cv2.bitwise_or(result,reference)
    
    return (np.sum(intersection)/np.sum(union))


def fileIdCatch(e):
    e=os.path.basename(e)
    e=e.split(".")
    e1=e[0].split("_")
    return int(e1[1])

def makeJson2Mask(dataset_dir,jsonFileName):
    mask_dir=os.path.join(dataset_dir, "label")
    image_dir=os.path.join(dataset_dir, "image")
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
        os.makedirs(image_dir)
    annotations = json.load(open(os.path.join(dataset_dir, jsonFileName)))
    annotations = list(annotations.values())  # don't need the dict keys
    #print(annotations)
    annotations = [a for a in annotations if a['regions']]
    #print(annotations)

    # Add images
    ix=0
    for a in annotations:
        polygons = [r['shape_attributes'] for r in a['regions'].values()]
        #polygons = [r['shape_attributes'] for r in a['regions']]
        name=a['filename']
        print(name)

        iname=str(ix)+".jpg"
        ix=ix+1
        mname=os.path.join(mask_dir, iname)
        print(mname)
        fname=os.path.join(image_dir, iname)
        print(fname)

        image_path = os.path.join(dataset_dir, name)
        image = skimage.io.imread(image_path)
        skimage.io.imsave(fname,image)
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        for i, p in enumerate(polygons):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc] = 255
        skimage.io.imsave(mname,mask)
