import json
from glob import glob
import os
import cv2

def img2jsonAnnEx(sourcePath, imgPatch, maskSource, maskPatch, maskLabel, showlist=True, fileName="via_region_data.json"):
    all_images = sorted(glob(os.path.join(sourcePath, '*'+imgPatch+'*')))
    all_masks = sorted(glob(os.path.join(maskSource, '*'+maskPatch+'*')))
    jsonF = images2json(all_images, all_masks, maskLabel, showlist)
    file = open(sourcePath+fileName, 'w')
    file.write(jsonF)
    file.close()
    print(sourcePath+fileName+" json ok")

    return True


def img2jsonAnnPre(sourcePath, imgPatch, maskSource, maskPatch, maskLabel, showlist=True, fileName="via_region_data.json"):
    all_images = sorted(glob(os.path.join(sourcePath, imgPatch+'*')))
    all_masks = sorted(glob(os.path.join(maskSource, maskPatch+'*')))
    jsonF = images2json(all_images, all_masks, maskLabel, showlist)
    file = open(sourcePath+fileName, 'w')
    file.write(jsonF)
    file.close()
    print(sourcePath+fileName+" json ok")

    return True


def images2json(all_images, all_masks, maskLabel, showlist=True):
    print(len(all_masks))
    for i in range(len(all_images)):
        fname = os.path.basename(all_images[i])
        mname = os.path.basename(all_masks[i])
        if (showlist):
            print("{} - {} / {}".format(i, fname, mname))
    data = {}
    i = 0
    for item in all_masks:
        img = all_images[i]
        sizeX = os.stat(img).st_size
        fname = os.path.basename(img)
        fnameA = fname.split("_")
        key = str(fname)+""+str(sizeX)
        value = {}
        value["fileref"] = ""
        value["size"] = sizeX
        value["filename"] = fname
        value["base64_img_data"] = ""
        fileattributes = {}
        value["file_attributes"] = fileattributes

        regions = {}
        shape_attributes = {}
        region_attributes = {}
        region_attributes["label"] = maskLabel

        src = cv2.imread(item)
        #print(np.histogram(src))

        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        # print(np.histogram(gray))
        blur = cv2.blur(gray, (3, 3))
        ret, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        cnt = sorted(contours, key=cv2.contourArea, reverse=False)
        # ROI will be object with biggest contour
        #mask = contours[0]
        #print(len(cnt))
        c1 = 0
        for item in cnt:
            xs = []
            ys = []
            cc = 0
            all_points_x = {}
            all_points_y = {}
            for ix in item:
                xs.append(ix[0][0])
                ys.append(ix[0][1])
                all_points_x[str(cc)] = ix[0][0]
                all_points_y[str(cc)] = ix[0][1]
                cc = cc+1
            dd = {}
            shape_attributes = {}
            shape_attributes["name"] = "polygon"
            xs.append(xs[0])
            ys.append(ys[0])
            shape_attributes["all_points_x"] = str(xs)
            shape_attributes["all_points_y"] = str(ys)
            dd["shape_attributes"] = shape_attributes
            dd["region_attributes"] = region_attributes

            regions[str(c1)] = dd
            c1 = c1+1

        value["regions"] = regions
        data[key] = value
        i = i+1

    json_data = json.dumps(data)
    json_data = json_data.replace('"[', '[')
    json_data = json_data.replace(']"', ']')
    json_data = json_data.replace('}}}}, "', '}}}}, \n "')
    return json_data
