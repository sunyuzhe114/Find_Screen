import numpy as np
import random
import cv2
import os
import sys
from matplotlib import pyplot as plt
import argparse
#导入math包
import math
import find_screen
def hammingDistance( x, y):
    #这里可能使用汉宁改进一下，识别为1，实际为0，可能是因为噪声引起，另外识别为0，实际为1可能是点打在了黑字上
    return bin(x^y).count('1')

def distance(p1,p2):
    x=p1[0]-p2[0]
    y=p1[1]-p2[1]
    #用math.sqrt（）求平方根
    return   math.sqrt((x**2)+(y**2))
#定义得到直线长度的函数

def toInt(content):
    ORG_POINT_COUNT=4
    value = 0
    for i in range(ORG_POINT_COUNT-1,-1,-1):
        for j in range(ORG_POINT_COUNT-1,-1,-1):
            value = value << 1
            #value += content[i][j] > 0 ? 1: 0;
            if(content[i][j]>0):
                value=value+1
    return value



def nothing(x):
    #print(x)
    pass

#检测图像中的点的函数
def detect_blob(im):
    global img_width,img_height,half_windowSize,beginX,beginY
    total_black_area=cv2.countNonZero(im)/(im.shape[0]*im.shape[1])
    #total_black_area=1
    #print('%.3f'%(total_black_area))
    params = cv2.SimpleBlobDetector_Params()

    params.filterByColor = 1
    params.blobColor=0
    # Change thresholds
    params.thresholdStep = 10
    params.minThreshold = 119
    params.maxThreshold = 143
    # params.minThreshold = 50
    # params.maxThreshold = 200
    #点最小间距
    params.minDistBetweenBlobs=30
    # Filter by Area.
    params.filterByArea = True
    #这个值要换算一下，针对 1080P的是3*3=9，如果图大于1080P要扩大面积
    params.minArea = 1#4,8均可
    params.maxArea = 81

    # Filter by CircularityThis just measures how close to a circle the blob is.
    # E.g. a regular hexagon has higher circularity than say a square.
    # To filter by circularity, set filterByCircularity = 1.
    # Then set appropriate values forminCircularity and maxCircularity.
    # Circularity is defined as (\frac{4*\pi*Area}{perimeter * perimeter}
    # 过滤斑点是否为圆
    params.filterByCircularity = True
    params.minCircularity = 0.4
    #params.maxCircularity = 1

    # Filter by Convexity # 过滤斑点是否为凸的
    #Convexity is defined as the (Area of the Blob / Area of it’s convex hull).
    # Now, Convex Hull of a shape is the tightest convex shape that completely encloses the shape.
    # To filter by convexity, set filterByConvexity = 1, followed by setting 0 ≤ minConvexity≤ 1 and maxConvexity ( ≤ 1)
    params.filterByConvexity = True
    params.minConvexity = 0.4
    #params.maxConvexity = 1

    # Filter by Inertia 惯量
    #for a circle, this value is 1, for an ellipse it is between 0 and 1,
    # and for a line it is 0. To filter by inertia ratio, set filterByInertia = 1,
    # and set 0 ≤ minInertiaRatio ≤ 1 and maxInertiaRatio (≤ 1 ) appropriately.
    params.filterByInertia = True
    params.minInertiaRatio = 0.1
    #params.maxInertiaRatio = 1

    # Create a detector with the parameters
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        detector = cv2.SimpleBlobDetector(params)
    else:
        detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(im)
   # print("detect points ",len(keypoints))
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
    # the size of the circle corresponds to the size of blob

    # f = open("result.csv", "w")
    # for p in keypoints:
    #     info=str(round(p.pt[0]))+"\t"+str(round(p.pt[1]))
    #     f.write(info+ '\n')
    #     #print(round(p.pt[0])," ",round(p.pt[1]))
    # f.close()

    im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


    crossPoints=[]

    for i in range(1,5):
        for j in range(1,5):
            crossPoints.append((round(i*half_windowSize*2/5),round(j*half_windowSize*2/5)))

            # cv2.circle(img_result, (round(i*half_windowSize*2/5),round(j*half_windowSize*2/5)) ,
            #               4,(0, 255, 0))

    array = [[0, 0, 0, 0], [0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0]]
    #for pt in keypoints:
    for i in range(len(keypoints)):
        for j in range(len(crossPoints)):
            # 获取两点之间直线的长度
            l = distance(keypoints[i].pt, crossPoints[j])
            if l< half_windowSize/DISTANCE_LIMIT :
                row =int(j/4)
                col =j%4
                array[col][row]=1
                #cv2.circle(img_result, crossPoints[j], 4,(0, 255, 0),thickness=-1)

    #将图片中黑色区过多的图片去除。
    if total_black_area<0.85:
        array = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    myresult =toInt(array)
    #print(array,hex(myresult))
    #print(hex(myresult))
    # f = open(checkpath+"result.txt", "a")
    # ##f.write(hex(myresult)+ '\n')
    # f.write(hex(myresult) + ',')
    # f.close()
    return im_with_keypoints,keypoints,myresult

def generate_date(img):

    #不同图片尺寸不一样，要动态算一下
    global img_width,img_height,half_windowSize,beginX,beginY
    img_width = img.shape[1]
    img_height = img.shape[0]
    half_windowSize = int(img_width / 12)
    beginX = half_windowSize
    beginY = half_windowSize

    gray_img = cv2.GaussianBlur(img, (3, 3), 0)  # 高斯滤波
    result=[]
    for j in range (0,10):
        if(beginY + half_windowSize + j * half_windowSize * 2) >=img.shape[0]:
            break
        for i in range(0,6):
            block_gray_img = gray_img[beginY - half_windowSize+j*half_windowSize*2:beginY + half_windowSize+j*half_windowSize*2,
                             beginX - half_windowSize+i*half_windowSize*2:beginX + half_windowSize+i*half_windowSize*2]
            ret, thresh_THRESH_OTSU = cv2.threshold(block_gray_img, 0, 255, cv2.THRESH_OTSU|cv2.THRESH_BINARY)
            #print(j,i,j*6+i)
            im_with_keypoints,keypoints,myresult=detect_blob(thresh_THRESH_OTSU)
            result.append(myresult)
    return result

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'gif', 'GIF'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def main(args):
    global checkpath,DISTANCE_LIMIT

    files=[]
    if (args.image!=None):
        checkpath= args.image.rsplit('/', 1)[0]+"/"
        files.append(args.image.rsplit('/', 1)[1])

    if (args.images_dir!= None):
        checkpath = args.images_dir
        # checkpath = r"new_2x2/"
        # checkpath = r"new_3x3/"
        # checkpath = r"ori_img/"
        #checkpath = r"new10/"
        # checkpath = r"bugs/"
        files = os.listdir(checkpath)

    imgPaths = files
    DISTANCE_LIMIT = 3 # halfwindowsize的3分之一长度做为点的偏移极限

    right_num = 0
    total_num = 0

    f = open(checkpath + 'keynames.txt', 'r')
    keyname = f.readlines()
    keyname = [x.strip() for x in keyname if x.strip()!='']
    f.close()
    f = open(checkpath + 'database.txt', 'r')
    fdatabases = f.readlines()  # 只读取1行
    f.close()

    for imgPath in imgPaths:
        if allowed_file(imgPath) == False:
            continue
        total_num += 1
        my_image = cv2.imread(checkpath + imgPath, cv2.IMREAD_GRAYSCALE)
        #这里将my_image从屏幕区中扣出，如果my_image是手机原图则不调用下面的话
        squares,my_image=find_screen.getFinalImage(my_image)

        os.makedirs(checkpath+"tmp", exist_ok=True)
        cv2.imwrite(checkpath+"tmp/_"+imgPath,my_image)
        chekdata=generate_date(my_image)

        findbest = -1
        minHanni = 960
        total_good_blocks=0
        #统计一下检测的有效换数，并换算成点数*16
        for i in range(0, 60):
            if (i >= len(chekdata) or chekdata[i] == ''):
                break
            if (chekdata[i] != 0):
                total_good_blocks += 1
        total_good_points=total_good_blocks *  16

        results=[]
        for j in range(len(fdatabases)):
            hndis = 0
            listdata = fdatabases[j].split(',')
            for i in range(0, 60):
                if (i >= len(chekdata) or chekdata[i] == ''):
                    break

                #if (i > 5 and i < 54 and i % 6 != 0 and (i - 5) % 6 != 0):  # 去除左上，左下，右上，右下的点
                    # if (i !=0 and i !=5 ):  # 去除左上，左下，右上，右下的点
                    # #hndis += hammingDistance(int(listdata[i], 16), int(chekdata[i], 16))
                if( chekdata[i]!=0):
                    hndis += hammingDistance(int(listdata[i], 16), chekdata[i])
            # print(keyname[j]," 汉宁距离",hndis ,"正确率",'%.3f' %((total_good_block*=16-hndis)/total_good_block*=16))
            results.append(hndis)
            if (hndis < minHanni):
                minHanni = hndis
                findbest = j
        test = np.array(results)
        top_k = test.argsort()[:5][::1]


        str_result={}
        for i in top_k:
           str_result[keyname[i]] = '%.2f'%((total_good_points-results[i])/total_good_points*100)+"%"

        if imgPath.find(keyname[findbest]) >= 0:
            right_num += 1
            print(imgPath, "内容为：", keyname[findbest], " 汉宁距离", minHanni, "正确率", '%.3f' % ((total_good_points - minHanni) / total_good_points),
                  "---正确")#,"可能结果",str_result)
        else:
            print(imgPath, "内容为：", keyname[findbest], " 汉宁距离", minHanni, "正确率", '%.3f' % ((total_good_points - minHanni) / total_good_points),
                  "---错误","可能结果",str_result)

    print("总数", total_num, "正确数量", right_num, "正确率", '%.3f' % (right_num / total_num))





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str,help="检测单张图片，例如: python img_process_demo.py --image c:/a.jpg",default=None)
    parser.add_argument('--images_dir', type=str, help="检测文件夹下所有图片，例如: python img_process_demo.py --images_dir c:/pic/",default=None)
    main_args = parser.parse_args()
    opt, unparsed = parser.parse_known_args()
    main(main_args)

    cv2.destroyAllWindows()
