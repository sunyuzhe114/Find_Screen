import numpy as np
import random
import cv2
import os
from matplotlib import pyplot as plt

#导入math包
import math


def distance(p1,p2):
    x=p1[0]-p2[0]
    y=p1[1]-p2[1]
    #用math.sqrt（）求平方根
    return   math.sqrt((x**2)+(y**2))
#定义得到直线长度的函数




def nothing(x):
    print(x)
    pass

def detect_blob(im):
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
    params.minArea = 8
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

    f = open("result.csv", "w")
    for p in keypoints:
        info=str(round(p.pt[0]))+"\t"+str(round(p.pt[1]))
        f.write(info+ '\n')
        #print(round(p.pt[0])," ",round(p.pt[1]))
    f.close()

    im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return im_with_keypoints,keypoints



path = r"new_3x3/"
files = os.listdir(path)
imgPaths=files
image_inedex=0
imgPath=imgPaths[image_inedex]

img = cv2.imread(path+imgPath, cv2.IMREAD_GRAYSCALE)


img_width=img.shape[1]
img_height=img.shape[0]
# cv2.namedWindow('o',0)
# cv2.resizeWindow("o", 768, 768)
# cv2.namedWindow('Keypoints', 0)
# cv2.resizeWindow("Keypoints", 768, 768)



half_windowSize =round(img_width/12)
print("windows size =",half_windowSize*2)
#half_windowSize=256
beginX=half_windowSize
beginY=half_windowSize
#不切分
img = img[ beginY - half_windowSize:beginY + half_windowSize,beginX - half_windowSize:beginX + half_windowSize ]
#img = cv2.imread("test.png", cv2.IMREAD_GRAYSCALE)
cv2.namedWindow("original_image", 0)
imgori = cv2.imread(path + imgPath)



#cv2.namedWindow('MEAN_C_adaptive',0)
cv2.namedWindow('GAUSSIAN_C_adaptive',0)
cv2.namedWindow('res',0)
cv2.resizeWindow("res", 512, 674)
#cv2.resizeWindow("MEAN_C_adaptive", 512, 512)
cv2.resizeWindow("GAUSSIAN_C_adaptive", 512, 512)
cv2.createTrackbar('area','res',15,64,nothing)
cv2.createTrackbar('min','res',128,255,nothing)
cv2.createTrackbar('max','res',255,255,nothing)
cur_flag = -1
while(1):
    #gray_img = img.copy()# 不滤波
    #gray_img=cv2.medianBlur(img,3) # 均值滤波
    ori_img_show=imgori.copy()
    gray_img = cv2.GaussianBlur(img, (3, 3), 0)#高斯滤波
    cv2.namedWindow("image",0)
    cv2.resizeWindow("image",512,512)
    cv2.imshow("image", img)

    cv2.imshow("gray_img", gray_img)
    # 获取键盘事件
    key = cv2.waitKey()

    # Esc，退出
    if key == 27:
        break
    # 判断是否按下其他键
    #if key > -1 and key != pre_flag:
    cur_flag = key


    # 响应事件
    if cur_flag == ord('n') :
        image_inedex=image_inedex+1
        if image_inedex== len(imgPaths):
                image_inedex=0
        imgPath = imgPaths[image_inedex]
        img = cv2.imread(path+imgPath, cv2.IMREAD_GRAYSCALE)
        imgori=cv2.imread(path+imgPath)



        img_width = img.shape[1]
        img_height = img.shape[0]
        img = img[beginY - half_windowSize:beginY + half_windowSize, beginX - half_windowSize:beginX + half_windowSize]


    #if cur_flag == ord('c') and key != pre_flag :
    if cur_flag == ord('c') :
        beginX=beginX+half_windowSize*2
        if beginX>=img_width:
            beginX = img_width -half_windowSize
        img = cv2.imread(path+imgPath, cv2.IMREAD_GRAYSCALE)
        img_width = img.shape[1]
        img_height = img.shape[0]
        img = img[beginY - half_windowSize:beginY + half_windowSize, beginX - half_windowSize:beginX + half_windowSize]

    #if cur_flag == ord('x') and key != pre_flag :
    if cur_flag ==  ord('x'):
        #if  pre_flag == -1:
        beginY=beginY+half_windowSize*2
        if beginY>=img_height:
            beginY = img_height -half_windowSize
        img = cv2.imread(path+imgPath, cv2.IMREAD_GRAYSCALE)
        img_width = img.shape[1]
        img_height = img.shape[0]
        img = img[beginY - half_windowSize:beginY + half_windowSize, beginX - half_windowSize:beginX + half_windowSize]


    if cur_flag == ord('z') :
        beginX=beginX-half_windowSize*2
        if(beginX- half_windowSize<0):
            beginX=half_windowSize
        img = cv2.imread(path+imgPath, cv2.IMREAD_GRAYSCALE)
        img_width = img.shape[1]
        img_height = img.shape[0]
        img = img[beginY - half_windowSize:beginY + half_windowSize, beginX - half_windowSize:beginX + half_windowSize]


    if cur_flag == ord('s') :
        beginY=beginY-half_windowSize*2
        if(beginY- half_windowSize<0):
            beginY=  half_windowSize
        img = cv2.imread(path+imgPath, cv2.IMREAD_GRAYSCALE)
        img = img[beginY - half_windowSize:beginY + half_windowSize, beginX - half_windowSize:beginX + half_windowSize]

    if cur_flag == ord('v') :
        cv2.imwrite("test.png",img)

    maxVal=cv2.getTrackbarPos('max','res')
    minVal=cv2.getTrackbarPos('min','res')
    areaVal = cv2.getTrackbarPos('area', 'res')
    ret, thresh1 = cv2.threshold(gray_img,minVal,maxVal, cv2.THRESH_BINARY)
    # ret, thresh2 = cv2.threshold(gray_img,minVal,maxVal, cv2.THRESH_BINARY_INV)
    # ret, thresh3 = cv2.threshold(gray_img,minVal,maxVal, cv2.THRESH_TRUNC)
    # ret, thresh4 = cv2.threshold(gray_img,minVal,maxVal, cv2.THRESH_TOZERO)
    # ret, thresh5 = cv2.threshold(gray_img,minVal,maxVal, cv2.THRESH_TOZERO_INV)
    ret, thresh_THRESH_OTSU = cv2.threshold(gray_img, minVal, maxVal, cv2.THRESH_OTSU|cv2.THRESH_BINARY)
    if(areaVal%2==0):
        areaVal=areaVal-1
    if(areaVal<=3):
        areaVal=3
    # MEAN_C_adaptive = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
    #                                         cv2.THRESH_BINARY, areaVal, 2)
    #MEAN_C_adaptive = cv2.medianBlur(MEAN_C_adaptive, 5)


    GAUSSIAN_C_adaptive = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                                cv2.THRESH_BINARY, areaVal, 2)

    GAUSSIAN_C_Blur =  cv2.medianBlur(GAUSSIAN_C_adaptive, 5)#高斯滤波
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    GAUSSIAN_C_Blur = cv2.dilate(GAUSSIAN_C_Blur, kernel, iterations=1)

    # Setup SimpleBlobDetector parameters.
    img_result,keypoints=detect_blob(thresh_THRESH_OTSU)
    #img_result = detect_blob(GAUSSIAN_C_Blur)
    #print("save image ")
    #cv2.imwrite("result.png",img_result)

#         for(int i=1;i<6;i++)
#             graphics.drawLine(i*180,0,i*180,1920);
#         for(int i=1;i<11;i++)
#             graphics.drawLine(0,i*180,1080,i*180);
    crossPoints=[]
    for i in range(1 , 5):
        cv2.line(img_result,pt1=(round(i*half_windowSize*2/5),0),
                 pt2=(round(i*half_windowSize*2/5),half_windowSize*2),color=(255,0,0))
        cv2.line(img_result, pt1=( 0,round(i * half_windowSize * 2 / 5)),
                 pt2=( half_windowSize * 2,round(i * half_windowSize * 2 / 5) ), color=(255, 0, 0))
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
            if l< half_windowSize/4 :
                row =int(j/4)
                col =j%4
                array[col][row]=1
                cv2.circle(img_result, crossPoints[j], 4,(0, 255, 0),thickness=-1)

    print(array)
    cv2.imshow('res',img_result)
    cv2.imshow('THRESH_BINARY', thresh1)
    # cv2.imshow('THRESH_BINARY_INV', thresh2)
    # cv2.imshow('THRESH_TRUNC', thresh3)
    # cv2.imshow('THRESH_TOZERO', thresh4)
    # cv2.imshow('THRESH_TOZERO_INV', thresh5)
    cv2.imshow('THRESH_OTSU', thresh_THRESH_OTSU)
    cv2.imshow('GAUSSIAN_C_adaptive', GAUSSIAN_C_adaptive)
    cv2.imshow('GAUSSIAN_C_Blur', GAUSSIAN_C_Blur)

    cv2.rectangle(ori_img_show, (beginX - half_windowSize, beginY - half_windowSize), (beginX + half_windowSize, beginY + half_windowSize),
                  (0, 0, 255),
                  4)

    cv2.imshow("original_image", ori_img_show)

cv2.destroyAllWindows()

