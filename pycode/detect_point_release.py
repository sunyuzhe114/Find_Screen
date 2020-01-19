import numpy as np
from time import *
import cv2
import argparse

import os
#导入math包
import math

import find_screen
bshow_debug_window=True

our_dict = {0x181: '0', 0x144: '1', 0x142: '2', 0x141: '3', 0x121: '4', 0x111: '5', 0x10c: '6', 0x10a: '7', 0x109: '8',
            0x105: '9', 0x103: 'a',  0xc4: 'b',  0xa1: 'c',  0x8c: 'd',  0x85: 'e',  0x64: 'f',  0x62: 'g',  0x61: 'h',
            0x54 : 'i' , 0x4c: 'j',  0x46: 'k',  0x45: 'l', 0x182: 'm', 0x150: 'n', 0x128: 'o', 0x118: 'p', 0x114: 'q',
            0x112: 'r',  0xc2: 's',  0x94: 't',  0x92: 'u',  0x91: 'v',  0x86: 'w',  0x83: 'x',  0x70: 'y',  0x68: 'z',
             0x52: 'A',  0x51: 'B',  0x38: 'C',  0x31: 'D',  0x2c: 'E',  0x29: 'F',  0x1c: ',',  0x15: '.',
            0x127: '起始帧', 0x4f: '中继帧',
            }
our_dict_index={0x181:  0,0x144:  1, 0x142:  2, 0x141:  3,0x121:4  , 0x111:5 , 0x10c:6 ,0x10a:7 , 0x109:8 , 0x105:9 ,0x103: 10,
                0xc4:  11,0xa1:  12,  0x8c: 13,  0x85: 14,0x64: 15 , 0x62: 16, 0x61: 17,0x54: 18, 0x4c: 19, 0x46: 20, 0x45: 21,
                0x182: 22,0x150: 23, 0x128: 24, 0x118: 25,0x114: 26,0x112: 27, 0xc2: 28,0x94: 29, 0x92: 30, 0x91: 31, 0x86: 32,
                0x83:  33,0x70:  34,  0x68: 35,  0x52: 36,0x51:37,    0x38:38, 0x31:39,  0x2c:40,  0x29:41,  0x1c:42,  0x15:43,
                0x127: 44,0x4f:  45,
         }

def hammingDistance( x, y):
    #这里可能使用汉宁改进一下，识别为1，实际为0，可能是因为噪声引起，另外识别为0，实际为1可能是点打在了黑字上
    return bin(x^y).count('1')

def distance(p1,p2):
    x=p1[0]-p2[0]
    y=p1[1]-p2[1]
    #用math.sqrt（）求平方根
    return   math.sqrt((x**2)+(y**2))


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

    if(len(keypoints)==0):
        return None,keypoints,0
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

def decode_data_block (img_data_block ):
    # 不同图片尺寸不一样，要动态算一下
    img_width = img_data_block.shape[1]
    img_height = img_data_block.shape[0]
    half_windowSize = int(img_width / 8)
    beginX = half_windowSize
    beginY = half_windowSize

    gray_img = img_data_block.copy()
    time_block_ID=[2,4,8]
    user_block_ID=[1,5,9,3,7,11]
    result = []
    for j in range(0, 3):
        for i in range(0, 4):
            block_gray_img = gray_img[
                             beginY - half_windowSize + j * half_windowSize * 2:beginY + half_windowSize + j * half_windowSize * 2,
                             beginX - half_windowSize + i * half_windowSize * 2:beginX + half_windowSize + i * half_windowSize * 2]
            ret, thresh_THRESH_OTSU = cv2.threshold(block_gray_img, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
            # print(j,i,j*6+i)
            im_with_keypoints, keypoints, myresult = detect_blob(thresh_THRESH_OTSU)
            result.append(myresult)
    username=[]
    time_data=[]
    for userid in user_block_ID:
        if our_dict.get(result[userid], 0) == 0:
            username.append('?')
            pass
        else:
            username.append(our_dict[result[userid]])
    for timeid in time_block_ID:
        if our_dict.get(result[timeid], 0) == 0:
            time_data.append('?')
            pass
        else:
            time_data.append( result[timeid] )


    print(username,time_data)
    return username,time_data



def generate_date(img ):
    begin_time = time()
    result = {}
    if img is None :
        result["result_username"] = 0
        result["username"] = '??????'
        result["result_usertime"] = 0
        result["usertime"] = [None, None, None]
        end_time = time()
        run_time = end_time - begin_time
        result["timecost"] = '%.1f' % run_time  # 该循环程序运行时间： 1.4201874732
        result["reason"]="load image error"
        print(result)
        return result

    #不同图片尺寸不一样，要动态算一下
    global img_width,img_height,half_windowSize,beginX,beginY,minVal,maxVal


    img_width = img.shape[1]
    img_height = img.shape[0]
    #half_windowSize = int(img_width / 12)
    #print("图片大小 ",img.shape)

    stepsize=20
    if(img_width==1080):
        half_windowSize = 40#
        stepsize=20
    else:# 后期这里要根据不同图片的大小来计算
        rate=img_width/1080
        half_windowSize = 40  *rate#
        stepsize = 20*rate

    beginX = half_windowSize
    beginY = half_windowSize

    #gray_img = cv2.GaussianBlur(img, (3, 3), 0)  # 高斯滤波
    user_list=[]
    time_list=[]
    gray_img= img.copy()


    jLoopNum=int((img_height-half_windowSize*2-1)/stepsize)
    iLoopNum=int((img_width-half_windowSize*2-1)/stepsize)
    b_exit=False
    for j in range (0,jLoopNum):
        if b_exit:
             break
        for i in range(0,iLoopNum):

            #block_gray_img = gray_img[beginY - half_windowSize+j*half_windowSize*2:beginY + half_windowSize+j*half_windowSize*2,
             #                 beginX - half_windowSize+i*half_windowSize*2:beginX + half_windowSize+i*half_windowSize*2]
            #(100, 640) 295 => 0x127 定义值=> 起始帧 距离 0.50
            block_gray_img = gray_img[int(j*stepsize) :int(j*stepsize + half_windowSize*2) ,
                               int(i *stepsize) :int(i *stepsize + half_windowSize*2 )]
            #block_gray_img=cv2.getRectSubPix(gray_img, (half_windowSize*2, half_windowSize*2), ( i *stepsize,j*stepsize ))

            #block_gray_img=cv2.cvtColor(block_gray_img,cv2.COLOR_BGR2GRAY)
            #ret, thresh_THRESH_OTSU = cv2.threshold(block_gray_img, 0, 255, cv2.THRESH_OTSU|cv2.THRESH_BINARY)
            ret, thresh_bin = cv2.threshold(block_gray_img, minVal, maxVal, cv2.THRESH_BINARY)
            #ori_img_show = imgori.copy()



            #cv2.waitKey(0)

            im_with_keypoints,keypoints,myresult=detect_blob(thresh_bin)
            if our_dict.get(myresult, 0) == 0:
                pass
            else:
                if(our_dict[myresult]=="起始帧"):

                    if bshow_debug_window is True:
                        cv2.rectangle(ori_img_show, (int(i *stepsize),int(j*stepsize)),
                                  (int(i *stepsize + half_windowSize*8), int(j*stepsize + half_windowSize*6)),
                                  (0, 0, 255),
                                  4)
                    data_block_img = gray_img[int(j * stepsize):int(j * stepsize + half_windowSize * 6),
                                     int(i * stepsize):int(i * stepsize + half_windowSize * 8)]
                    user_name,user_time=decode_data_block(data_block_img)
                    user_list.append(user_name)
                    time_list.append(user_time)


                if (our_dict[myresult] == "中继帧"):
                    if bshow_debug_window is True:
                        cv2.rectangle(ori_img_show, (int(i *stepsize- half_windowSize*4),int(j*stepsize-  half_windowSize*2)),
                                  (int(i *stepsize + half_windowSize*4),int( j*stepsize + half_windowSize*4)),
                                  (0, 0, 255),
                                  4)
                    #这晨如果检测错误，会取出异常区
                    data_block_img = gray_img[int(j * stepsize-  half_windowSize*2):int(j * stepsize + half_windowSize * 4),
                                     int(i * stepsize- half_windowSize*4):int(i * stepsize + half_windowSize * 4)]
                    user_name, user_time = decode_data_block(data_block_img)
                    user_list.append(user_name)
                    time_list.append(user_time)
                    # b_exit=True
                    # break

                    if bshow_debug_window is True:
                        cv2.imshow("image", block_gray_img)
                        cv2.imshow("original_image", ori_img_show)
                        print("坐标",(i*stepsize , j *stepsize),myresult, "=>", hex(myresult), "定义值=>", our_dict[myresult] ,"block",data_block_img.shape)
                        cv2.imshow("THRESH_BINARY", data_block_img)
                        cv2.waitKey(0)
                    pass
            #result.append(myresult)
    #print(user_list,time_list)

    user_name_T = list(map(list, zip(*user_list)))
    user_time_T = list(map(list, zip(*time_list)))

    final_user_name_list=[]
    final_user_time_list=[]
    result = {}
    if len(user_name_T)==0:
        result["result_username"] = 0
        result["username"] = '??????'
        result["result_usertime"] = 0
        result["usertime"] = [None,None,None]
        end_time = time()
        run_time = end_time - begin_time
        result["reason"] = "final_user_name_list =NULL"
        result["timecost"] = '%.1f' % run_time  # 该循环程序运行时间： 1.4201874732
        return result
    else:
        for i in range(0,6):
            final_user_name_list.append(max_count(user_name_T[i]))

    for i in range(0, 3):
        final_user_time_list.append(max_count(user_time_T[i]))

    final_str_username=''.join(str(i) for i in final_user_name_list)
    #这里要把时间 decode一下

    if(final_str_username.find("None")!=-1):
        errornum=final_str_username.count('None')
        result["result_username"] = '%.2f'%((6-errornum)/6)
        final_str_username=final_str_username.replace("None","?")
    else:
        result["result_username"] = 1

    result["username"] = final_str_username

    final_str_time=''
    time_index = final_user_time_list.index(None) if None in final_user_time_list else -1
    if(time_index!=-1):
        errornum = final_user_time_list.count(None)
        result["result_usertime"] =  '%.2f'%((3-errornum)/3)
        final_str_time=str(final_user_time_list)
    else:
        result["result_usertime"] = 1
        final_str_time = decodetime(final_user_time_list)
    result["usertime"] = final_str_time
    result["reason"] = ""
    end_time = time()
    run_time = end_time - begin_time
    result["timecost"]= '%.1f'%run_time   # 该循环程序运行时间： 1.4201874732
    #final_str_usertime = ' '.join(str(i) for i in final_user_name_list)
    print(result)

    return result

def max_count(lt):
    # 定义一个字典，用于存放元素及出现的次数
    d = {}
    # 记录最大的次数的元素
    max_key = None
    # 遍历列表，统计每个元素出现的次数，然后保存到字典中
    for i in lt:
        if i not in d and i!='?':
            # 计算元素出现的次数
            count = lt.count(i)
            # 保存到字典中
            d[i] = count
            # 记录次数最大的元素
            if count > d.get(max_key, 0):
                max_key = i
    return max_key


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
def decodetime(final_user_time_list):
    timeData=0
    for i in range(0,3):
        #timeData += timeData * 44 + final_user_time_list_code[i]
        timeData = timeData * 44 + our_dict_index[final_user_time_list[i]]


    hour = timeData % 24
    timeData=int(timeData /  24)
    day = int(timeData % 31) + 1
    timeData=int(timeData /  31)
    month = timeData % 12 + 1
    year = int(timeData / 12) + 2020
    strTime=str(year)+"-"+str(month)+"-"+str(day)+" "+str(hour)+"h"
    return  strTime



def main(args):

    global checkpath,imgPath,DISTANCE_LIMIT ,half_windowSize,ori_img_show,beginX,beginY,\
        img_width,img_height,imgori,img,zoom,maxVal,minVal,bshow_debug_window


    checkpath = ''
    DISTANCE_LIMIT = 5
    #二值化最大与最小值
    cv2.namedWindow("original_image", 0)
    cv2.setMouseCallback('original_image', on_mouse_original_image)
    cv2.resizeWindow('original_image', 540, 960)
    cv2.namedWindow('res', 0)
    cv2.resizeWindow("res", 512, 674)

    cv2.namedWindow('THRESH_BINARY', 0)
    cv2.resizeWindow('THRESH_BINARY', 512, 512)
    cv2.namedWindow('THRESH_OTSU', 0)
    cv2.resizeWindow('THRESH_OTSU', 512, 512)
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


    maxVal = 255
    minVal = 128
    files = []
    if (args.image != None):
        checkpath = args.image.rsplit('/', 1)[0] + "/"
        files.append(args.image.rsplit('/', 1)[1])

    #这里进行图片测试
    for imgPath in files:
        if allowed_file(imgPath) == False:
            continue

        my_image = cv2.imread(checkpath + imgPath, cv2.IMREAD_GRAYSCALE)

        # squares, my_image = find_screen.getFinalImage(my_image)
        # os.makedirs(checkpath + "tmp", exist_ok=True)
        # cv2.imwrite(checkpath + "tmp/_" + imgPath, my_image)
        imgori = my_image.copy()
        ori_img_show = my_image.copy()
        generate_date(my_image)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help="检测单张图片，例如: python img_process_demo.py --image 1x1/ori/720.png",
                        default=None)
    main_args = parser.parse_args()
    opt, unparsed = parser.parse_known_args()
    main(main_args)
    #开放8100端口


