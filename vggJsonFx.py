import json
from glob import glob
import cv2
import skimage, os
import numpy as np # linear algebra
import pandas as pd
from skimage import img_as_ubyte
import skimage.io
import skimage.draw

def img2jsonAnnEx(sourcePath,imgPatch,maskSource,maskPatch,maskLabel,showlist=True,fileName="via_region_data.json"):
    all_images =sorted(glob(os.path.join(sourcePath,'*'+imgPatch+'*')))
    all_masks  =sorted(glob(os.path.join(maskSource,'*'+maskPatch+'*')))
    jsonF=images2json(all_images,all_masks,maskLabel,showlist)
    file = open(sourcePath+fileName, 'w') 
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
        blur = cv2.blur(gray, (5, 5))
        ret, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
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

        iname=name
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
