import numpy as np
import random
import cv2
import os
from matplotlib import pyplot as plt
import argparse
#导入math包
import math
import find_screen
from datetime import datetime
our_dict={0x181:'0',
0x144:'1',
0x142:'2',
0x141:'3',
0x121:'4',
0x111:'5',
0x10c:'6',
0x10a:'7',
0x109:'8',
0x105:'9',
0x103: 'a',
0xc4: 'b',
0xa1: 'c',
0x8c: 'd',
0x85: 'e',
0x64: 'f',
0x62: 'g',
0x61: 'h',
0x54: 'i',
0x4c: 'j',
0x46: 'k',
0x45: 'l',
0x182: 'm',
0x150: 'n',
0x128: 'o',
0x118: 'p',
0x114: 'q',
0x112: 'r',
0xc2: 's',
0x94: 't',
0x92: 'u',
0x91: 'v',
0x86: 'w',
0x83: 'x',
0x70: 'y',
0x68: 'z',
0x52:'A',
0x51:'B',
0x38:'C',
0x31:'D',
0x2c:'E',
0x29:'F',
0x1c:',',
0x15:'.',
0x127:'起始帧',
0x1c9:'中继帧',#0x4f:'中继帧',
         }
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
    ORG_POINT_COUNT=3
    value = 0
    # for i in range(ORG_POINT_COUNT-1,-1,-1):
    #     for j in range(ORG_POINT_COUNT-1,-1,-1):
    for i in range(0,ORG_POINT_COUNT):
        for j in range (0,ORG_POINT_COUNT):
            value = value << 1

            if(content[i][j]>0):
                value=value+1
    return value



def nothing(x):
    #print(x)
    pass

#检测图像中的点的函数
def detect_blob(im):
    global img_width,img_height,half_windowSize,beginX,beginY
    #total_black_area=cv2.countNonZero(im)/(im.shape[0]*im.shape[1])
    #total_black_area=1
    #print('%.3f'%(total_black_area))
    params = cv2.SimpleBlobDetector_Params()

    params.filterByColor = 1
    params.blobColor=0
    # Change thresholds
    params.thresholdStep = 1
    # params.minThreshold = 119
    # params.maxThreshold = 143
    params.minThreshold = 119
    params.maxThreshold = 170
    #点最小间距
    params.minDistBetweenBlobs=5#30
    # Filter by Area.
    params.filterByArea = True
    #这个值要换算一下，针对 1080P的是3*3=9，如果图大于1080P要扩大面积
    params.minArea = 1#4,8均可
    params.maxArea = 16

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
    params.minConvexity = 0.1
    #params.maxConvexity = 1

    # Filter by Inertia 惯量
    #for a circle, this value is 1, for an ellipse it is between 0 and 1,
    # and for a line it is 0. To filter by inertia ratio, set filterByInertia = 1,
    # and set 0 ≤ minInertiaRatio ≤ 1 and maxInertiaRatio (≤ 1 ) appropriately.
    params.filterByInertia = True
    params.minInertiaRatio = 0.01
    #params.maxInertiaRatio = 1

    # Create a detector with the parameters
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

    for i in range(1,4):
        for j in range(1,4):
            crossPoints.append((round(i*half_windowSize*2/4),round(j*half_windowSize*2/4)))

            # cv2.circle(img_result, (round(i*half_windowSize*2/5),round(j*half_windowSize*2/5)) ,
            #               4,(0, 255, 0))

    array = [[0, 0, 0 ], [0, 0, 0 ],[0, 0, 0 ]]
    #for pt in keypoints:
    for i in range(len(keypoints)):
        for j in range(len(crossPoints)):
            # 获取两点之间直线的长度
            l = distance(keypoints[i].pt, crossPoints[j])
            if l< half_windowSize/DISTANCE_LIMIT :
                row =int(j/3)
                col =j%3
                array[col][row]=1
                #cv2.circle(img_result, crossPoints[j], 4,(0, 255, 0),thickness=-1)

    #将图片中黑色区过多的图片去除。
    # if total_black_area<0.85:
    #     array = [[0, 0, 0 ], [0, 0, 0 ],[0, 0, 0 ]]

    myresult =toInt(array)
    #print(array,hex(myresult))
    #print(hex(myresult))
    # f = open(checkpath+"result.txt", "a")
    # ##f.write(hex(myresult)+ '\n')
    # f.write(hex(myresult) + ',')
    # f.close()
    return im_with_keypoints,keypoints,myresult

def generate_date(img ):
    #return

    #不同图片尺寸不一样，要动态算一下
    global img_width,img_height,half_windowSize,beginX,beginY,minVal,maxVal

    img_width = img.shape[1]
    img_height = img.shape[0]
    #half_windowSize = int(img_width / 12)
    half_windowSize = 40#
    beginX = half_windowSize
    beginY = half_windowSize

    #gray_img = cv2.GaussianBlur(img, (3, 3), 0)  # 高斯滤波
    gray_img= img.copy()
    result=[]
    stepsize=20
    jLoopNum=int((img_height-half_windowSize*2-1)/stepsize)
    iLoopNum=int((img_width-half_windowSize*2-1)/stepsize)
    for j in range (0,jLoopNum):
        # if(beginY + half_windowSize + j * half_windowSize * 2) >=img.shape[0]:
        #     break
        for i in range(0,iLoopNum):

            #block_gray_img = gray_img[beginY - half_windowSize+j*half_windowSize*2:beginY + half_windowSize+j*half_windowSize*2,
             #                 beginX - half_windowSize+i*half_windowSize*2:beginX + half_windowSize+i*half_windowSize*2]
            #(100, 640) 295 => 0x127 定义值=> 起始帧 距离 0.50
            block_gray_img = gray_img[j*stepsize :j*stepsize + half_windowSize*2 ,
                               i *stepsize :i *stepsize + half_windowSize*2 ]
            #block_gray_img=cv2.getRectSubPix(gray_img, (half_windowSize*2, half_windowSize*2), ( i *stepsize,j*stepsize ))

            #block_gray_img=cv2.cvtColor(block_gray_img,cv2.COLOR_BGR2GRAY)
            #ret, thresh_THRESH_OTSU = cv2.threshold(block_gray_img, 0, 255, cv2.THRESH_OTSU|cv2.THRESH_BINARY)
            ret, thresh_bin = cv2.threshold(block_gray_img, minVal, maxVal, cv2.THRESH_BINARY)
            ori_img_show = imgori.copy()



            #cv2.waitKey(0)

            im_with_keypoints,keypoints,myresult=detect_blob(thresh_bin)
            if our_dict.get(myresult, 0) == 0:
                pass
            else:
                cv2.rectangle(ori_img_show, (i *stepsize,j*stepsize),
                          (i *stepsize + half_windowSize*2, j*stepsize + half_windowSize*2),
                          (0, 0, 255),
                          4)
                cv2.imshow("image", block_gray_img)
                cv2.imshow("original_image", ori_img_show)
                print("坐标",(i*stepsize , j *stepsize),myresult, "=>", hex(myresult), "定义值=>", our_dict[myresult] )
                cv2.imshow("THRESH_BINARY", thresh_bin)
                cv2.waitKey(0)
            #result.append(myresult)
    return result

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'gif', 'GIF'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
# mouse callback function
def on_mouse_original_image(event,x,y,flags,param):
    global beginX,beginY,half_windowSize,img_width,img_height
    if event == cv2.EVENT_LBUTTONDOWN:
        #print("beginX",beginX,"beginY",beginY)
        beginX=x
        beginY=y
        if beginX >= img_width:
            beginX = img_width - half_windowSize
        if beginY >= img_height:
            beginY = img_height - half_windowSize
        if (beginX - half_windowSize < 0):
            beginX = half_windowSize

        if (beginY - half_windowSize < 0):
            beginY = half_windowSize
    zoom = cv2.getRectSubPix(imgori, (128, 128), (x + 0.5, y + 0.5))
    zoom = cv2.pyrUp(zoom)
    zoom = cv2.pyrUp(zoom)
    zoom = cv2.pyrUp(zoom)
    cv2.imshow('zoom', zoom)
def on_mouse_grey_image(event,x,y,flags,param):
    global beginX,beginY,half_windowSize,img_width,img_height,zoom
    if event == cv2.EVENT_LBUTTONDOWN:
        #print("beginX",beginX,"beginY",beginY)
        zoom = cv2.getRectSubPix(img, (7, 7), (x + 0.5, y + 0.5))
        #print(zoom.shape)
        cv2.imshow('zoom', zoom)

def hist_lines(im):
    h = np.zeros((300,256,3))
    if len(im.shape)!=2:
        print("hist_lines applicable only for grayscale images")
        #print("so converting image to grayscale for representation"
        im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    hist_item = cv2.calcHist([im],[0],None,[256],[0,256])
    cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
    hist=np.int32(np.around(hist_item))
    for x,y in enumerate(hist):
        cv2.line(h,(x,0),(x,y),(255,255,255))
    y = np.flipud(h)
    return y

def main(args):
    global checkpath,imgPath,DISTANCE_LIMIT ,half_windowSize,ori_img_show,beginX,beginY,\
        img_width,img_height,imgori,img,zoom,maxVal,minVal
    #checkpath = r"new_2x2/"
    # checkpath = r"new_3x3/"
    #checkpath = r"ori_img/"
    #checkpath = r"new10/"
    checkpath = r"1x1/ori/"
    #checkpath = r"1x1/test/"
    files = os.listdir(checkpath)
    imgPaths = files
    image_inedex = 0
    imgPath = imgPaths[image_inedex]
    DISTANCE_LIMIT = 5 # halfwindowsize的3分之一长度做为点的偏移极限


    img = cv2.imread(checkpath + imgPath, cv2.IMREAD_GRAYSCALE)

    img_width = img.shape[1]
    img_height = img.shape[0]
    # cv2.namedWindow('o',0)
    # cv2.resizeWindow("o", 768, 768)
    # cv2.namedWindow('Keypoints', 0)
    # cv2.resizeWindow("Keypoints", 768, 768)

    #half_windowSize = round(img_width / 48)
    half_windowSize = 40 #round(img_width / 25)#这个值要根据图像大小来估算

    print("half_windowSize size =", half_windowSize )

    beginX = half_windowSize
    beginY = half_windowSize


    # 不切分
   # img = img[beginY - half_windowSize:beginY + half_windowSize, beginX - half_windowSize:beginX + half_windowSize]
    # img = cv2.imread("test.png", cv2.IMREAD_GRAYSCALE)


    cv2.namedWindow("original_image", 0)
    cv2.setMouseCallback('original_image', on_mouse_original_image)
    imgori = cv2.imread(checkpath + imgPath)


    cv2.namedWindow('res', 0)
    cv2.resizeWindow("res", 512, 674)

    cv2.namedWindow('THRESH_BINARY', 0)
    cv2.resizeWindow( 'THRESH_BINARY',  512, 512)
    cv2.namedWindow('THRESH_OTSU', 0)
    cv2.resizeWindow( 'THRESH_OTSU',  512, 512)
    # cv2.namedWindow('GAUSSIAN_C_adaptive', 0)
    # cv2.resizeWindow( 'GAUSSIAN_C_adaptive',  512, 512)
    # cv2.namedWindow('GAUSSIAN_C_Blur', 0)
    # cv2.resizeWindow( 'GAUSSIAN_C_Blur',  512, 512)

    cv2.namedWindow("zoom", 0)
    cv2.resizeWindow('zoom', 512, 512)
    cv2.namedWindow("image", 0)
    cv2.resizeWindow("image", 512, 512)
    cv2.setMouseCallback('image', on_mouse_grey_image)
    cv2.createTrackbar('area', 'res', 15, 64, nothing)
    cv2.createTrackbar('min', 'res', 128, 255, nothing)
    cv2.createTrackbar('max', 'res', 255, 255, nothing)
    cur_flag = -1

    maxVal = cv2.getTrackbarPos('max', 'res')
    minVal = cv2.getTrackbarPos('min', 'res')
    areaVal = cv2.getTrackbarPos('area', 'res')

    #这里进行图片测试
    generate_date(imgori.copy())

    while (1):
        # gray_img = img.copy()# 不滤波
        # gray_img=cv2.medianBlur(img,3) # 均值滤波
        ori_img_show = imgori.copy()
        #gray_img = cv2.GaussianBlur(img, (3, 3), 0)  # 高斯滤波
        #gray_img = img.copy()

        cv2.imshow("image", img)

        #cv2.imshow("gray_img", gray_img)
        # 获取键盘事件
        key = cv2.waitKey()

        # Esc，退出
        if key == 27:
            break
        # 判断是否按下其他键
        # if key > -1 and key != pre_flag:
        cur_flag = key

        # 响应事件
        if cur_flag == ord('n'):
            image_inedex = image_inedex + 1
            if image_inedex == len(imgPaths):
                image_inedex = 0
            imgPath = imgPaths[image_inedex]
            img = cv2.imread(checkpath + imgPath, cv2.IMREAD_GRAYSCALE)
            imgori = cv2.imread(checkpath + imgPath)
            img_width = img.shape[1]
            img_height = img.shape[0]
            img = img[beginY - half_windowSize:beginY + half_windowSize,
                  beginX - half_windowSize:beginX + half_windowSize]
            print("change file to ", imgPath)



        if cur_flag == ord('c'):
            beginX = beginX + 1
            if beginX >= img_width:
                beginX = img_width - half_windowSize
            img = cv2.imread(checkpath + imgPath, cv2.IMREAD_GRAYSCALE)
            img_width = img.shape[1]
            img_height = img.shape[0]
            img = img[beginY - half_windowSize:beginY + half_windowSize,
                  beginX - half_windowSize:beginX + half_windowSize]

            # if cur_flag == ord('x') and key != pre_flag :
        if cur_flag == ord('x'):
            # if  pre_flag == -1:
            beginY = beginY + 1
            if beginY >= img_height:
                beginY = img_height - half_windowSize
            img = cv2.imread(checkpath + imgPath, cv2.IMREAD_GRAYSCALE)
            img_width = img.shape[1]
            img_height = img.shape[0]
            img = img[beginY - half_windowSize:beginY + half_windowSize,
                  beginX - half_windowSize:beginX + half_windowSize]

        if cur_flag == ord('z'):
            beginX = beginX - 1
            if (beginX - half_windowSize < 0):
                beginX = half_windowSize
            img = cv2.imread(checkpath + imgPath, cv2.IMREAD_GRAYSCALE)
            img_width = img.shape[1]
            img_height = img.shape[0]
            img = img[beginY - half_windowSize:beginY + half_windowSize,
                  beginX - half_windowSize:beginX + half_windowSize]

        if cur_flag == ord('s'):
            beginY = beginY - 1
            if (beginY - half_windowSize < 0):
                beginY = half_windowSize
            img = cv2.imread(checkpath + imgPath, cv2.IMREAD_GRAYSCALE)
            img = img[beginY - half_windowSize:beginY + half_windowSize,
                  beginX - half_windowSize:beginX + half_windowSize]

        if cur_flag == ord(' '):

            img = cv2.imread(checkpath + imgPath, cv2.IMREAD_GRAYSCALE)
            img = img[beginY - half_windowSize:beginY + half_windowSize,
                  beginX - half_windowSize:beginX + half_windowSize]

        if cur_flag == ord('v'):
            now  =datetime.now()  # 格式为 datetime.datetime
            now_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')# 格式为 str

            cv2.imwrite("./blob_img/blob"+now_time+".png", zoom)
            print("save ","./blob_img/blob"+now_time+".png")

        #gray_img = cv2.GaussianBlur(img, (3, 3), 0)  # 高斯滤波

        # 灰度拉伸
        #hist_full = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
        gray_img = img.copy()
        #lines = hist_lines(gray_img)
        #cv2.imshow('gray_hist_ori', lines)

        # plt.plot(hist_full)
        # plt.show()
        # Imax = np.max(gray_img)
        # Imin = np.min(gray_img)
        # MAX = 255
        # MIN = 0
        #gray_img_ext = (gray_img - Imin) / (Imax - Imin) * (MAX - MIN) + MIN
        #gray_img_ext = cv2.equalizeHist(gray_img)

        #lines = hist_lines(gray_img_ext.astype("uint8"))
        #cv2.imshow('gray_hist_ext', lines)

        #gray_img=gray_img_ext.astype("uint8").copy()
        #hist_full = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
        #cv2.imshow("gray_hist_ext",hist_full)
        # plt.plot(hist_full)
        # plt.show()

        maxVal = cv2.getTrackbarPos('max', 'res')
        minVal = cv2.getTrackbarPos('min', 'res')
        areaVal = cv2.getTrackbarPos('area', 'res')
        ret, thresh_bin = cv2.threshold(gray_img, minVal, maxVal, cv2.THRESH_BINARY)
        # ret, thresh2 = cv2.threshold(gray_img,minVal,maxVal, cv2.THRESH_BINARY_INV)
        # ret, thresh3 = cv2.threshold(gray_img,minVal,maxVal, cv2.THRESH_TRUNC)
        # ret, thresh4 = cv2.threshold(gray_img,minVal,maxVal, cv2.THRESH_TOZERO)
        # ret, thresh5 = cv2.threshold(gray_img,minVal,maxVal, cv2.THRESH_TOZERO_INV)
        ret, thresh_THRESH_OTSU = cv2.threshold(gray_img, minVal, maxVal, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
        if (areaVal % 2 == 0):
            areaVal = areaVal - 1
        if (areaVal <= 3):
            areaVal = 3
        # MEAN_C_adaptive = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
        #                                         cv2.THRESH_BINARY, areaVal, 2)
        # MEAN_C_adaptive = cv2.medianBlur(MEAN_C_adaptive, 5)

        GAUSSIAN_C_adaptive = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                                    cv2.THRESH_BINARY, areaVal, 2)

        GAUSSIAN_C_Blur = cv2.medianBlur(GAUSSIAN_C_adaptive, 3)  # 高斯滤波
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        GAUSSIAN_C_Blur = cv2.dilate(GAUSSIAN_C_Blur, kernel, iterations=1)


        # Setup SimpleBlobDetector parameters.
        #img_result, keypoints,myresult = detect_blob(thresh_THRESH_OTSU)
        #img_result, keypoints,myresult = detect_blob(thresh1)
        img_result, keypoints, myresult = detect_blob(thresh_bin)
        # print("save image ")
        # cv2.imwrite("result.png",img_result)

        crossPoints = []
        for i in range(1, 4):
            cv2.line(img_result, pt1=(round(i * half_windowSize * 2 / 4), 0),
                     pt2=(round(i * half_windowSize * 2 / 4), half_windowSize * 2), color=(255, 0, 0))
            cv2.line(img_result, pt1=(0, round(i * half_windowSize * 2 / 4)),
                     pt2=(half_windowSize * 2, round(i * half_windowSize * 2 / 4)), color=(255, 0, 0))
        for i in range(1, 4):
            for j in range(1, 4):
                crossPoints.append((round(i * half_windowSize * 2 / 4), round(j * half_windowSize * 2 / 4)))

                # cv2.circle(img_result, (round(i*half_windowSize*2/5),round(j*half_windowSize*2/5)) ,
                #               4,(0, 255, 0))

        array = [[0, 0, 0 ], [0, 0, 0 ], [0, 0, 0 ] ]
        # for pt in keypoints:
        totalDis=0
        for i in range(len(keypoints)):
            for j in range(len(crossPoints)):
                # 获取两点之间直线的长度
                l = distance(keypoints[i].pt, crossPoints[j])
                if l < half_windowSize / DISTANCE_LIMIT:
                    row = int(j / 3)
                    col = j % 3
                    array[col][row] = 1
                    cv2.circle(img_result, crossPoints[j], 2, (0, 255, 0), thickness=-1)
                    totalDis+=l

        #print(array)
        myresult =toInt(array)
        if our_dict.get(myresult, 0) == 0:
            pass
        else:
            print((beginX,beginY),thresh_bin.shape,myresult, "=>",hex(myresult),"定义值=>",our_dict[myresult],"距离",'%.2f'%totalDis)


        cv2.imshow('res', img_result)
        cv2.imshow('THRESH_BINARY', thresh_bin)
        # cv2.imshow('THRESH_BINARY_INV', thresh2)
        # cv2.imshow('THRESH_TRUNC', thresh3)
        # cv2.imshow('THRESH_TOZERO', thresh4)
        # cv2.imshow('THRESH_TOZERO_INV', thresh5)
        cv2.imshow('THRESH_OTSU', thresh_THRESH_OTSU)
        #cv2.imshow('GAUSSIAN_C_adaptive', GAUSSIAN_C_adaptive)
        #cv2.imshow('GAUSSIAN_C_Blur', GAUSSIAN_C_Blur)

        ori_img_show = imgori.copy()

        cv2.rectangle(ori_img_show, (beginX - half_windowSize, beginY - half_windowSize),
                      (beginX + half_windowSize, beginY + half_windowSize),
                      (0, 0, 255),
                      4)

        cv2.imshow("original_image", ori_img_show)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='saved_models/stegastamp_pretrained')
    parser.add_argument('--image', type=str, default='encoded_img/lALPDgQ9rbLnuyZWVQ_85_86.png')
    parser.add_argument('--images_dir', type=str, default=None)
    parser.add_argument('--secret_size', type=int, default=100)
    main_args = parser.parse_args()

    main(main_args)


