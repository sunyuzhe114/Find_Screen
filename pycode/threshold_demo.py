import numpy as np
import random
import cv2
import os
from matplotlib import pyplot as plt




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
    params.minInertiaRatio = 0.4
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
    return im_with_keypoints



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




windowSize=256
beginX=windowSize+ 50
beginY=windowSize+ 50
#不切分
img = img[ beginY - windowSize:beginY + windowSize,beginX - windowSize:beginX + windowSize ]
#img = cv2.imread("test.png", cv2.IMREAD_GRAYSCALE)
cv2.namedWindow("original_image", 0)
imgori = cv2.imread(path + imgPath)



#cv2.namedWindow('MEAN_C_adaptive',0)
cv2.namedWindow('GAUSSIAN_C_adaptive',0)
cv2.namedWindow('res',0)
cv2.resizeWindow("res", 512, 674)
#cv2.resizeWindow("MEAN_C_adaptive", 512, 512)
cv2.resizeWindow("GAUSSIAN_C_adaptive", 512, 512)
cv2.createTrackbar('area','res',3,64,nothing)
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
        img = img[beginY - windowSize:beginY + windowSize, beginX - windowSize:beginX + windowSize]


    #if cur_flag == ord('c') and key != pre_flag :
    if cur_flag == ord('c') :
        beginX=beginX+windowSize
        if beginX>=img_width:
            beginX = img_width -windowSize
        img = cv2.imread(path+imgPath, cv2.IMREAD_GRAYSCALE)
        img_width = img.shape[1]
        img_height = img.shape[0]
        img = img[beginY - windowSize:beginY + windowSize, beginX - windowSize:beginX + windowSize]

    #if cur_flag == ord('x') and key != pre_flag :
    if cur_flag ==  ord('x'):
        #if  pre_flag == -1:
        beginY=beginY+windowSize
        if beginY>=img_height:
            beginY = img_height -windowSize
        img = cv2.imread(path+imgPath, cv2.IMREAD_GRAYSCALE)
        img_width = img.shape[1]
        img_height = img.shape[0]
        img = img[beginY - windowSize:beginY + windowSize, beginX - windowSize:beginX + windowSize]


    if cur_flag == ord('z') :
        beginX=beginX-windowSize
        if(beginX- windowSize<0):
            beginX=windowSize
        img = cv2.imread(path+imgPath, cv2.IMREAD_GRAYSCALE)
        img_width = img.shape[1]
        img_height = img.shape[0]
        img = img[beginY - windowSize:beginY + windowSize, beginX - windowSize:beginX + windowSize]


    if cur_flag == ord('s') :
        beginY=beginY-windowSize
        if(beginY- windowSize<0):
            beginY=  windowSize
        img = cv2.imread(path+imgPath, cv2.IMREAD_GRAYSCALE)
        img = img[beginY - windowSize:beginY + windowSize, beginX - windowSize:beginX + windowSize]

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
    img_result=detect_blob(thresh_THRESH_OTSU)
    #img_result = detect_blob(GAUSSIAN_C_Blur)
    #print("save image ")
    #cv2.imwrite("result.png",img_result)


    cv2.imshow('res',img_result)
    cv2.imshow('THRESH_BINARY', thresh1)
    # cv2.imshow('THRESH_BINARY_INV', thresh2)
    # cv2.imshow('THRESH_TRUNC', thresh3)
    # cv2.imshow('THRESH_TOZERO', thresh4)
    # cv2.imshow('THRESH_TOZERO_INV', thresh5)
    cv2.imshow('THRESH_OTSU', thresh_THRESH_OTSU)
    cv2.imshow('GAUSSIAN_C_adaptive', GAUSSIAN_C_adaptive)
    cv2.imshow('GAUSSIAN_C_Blur', GAUSSIAN_C_Blur)
    cv2.rectangle(ori_img_show, (beginX - windowSize, beginY - windowSize), (beginX + windowSize, beginY + windowSize),
                  (0, 0, 255),
                  4)
    cv2.imshow("original_image", ori_img_show)
cv2.destroyAllWindows()

