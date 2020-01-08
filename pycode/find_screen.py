
#!/usr/bin/env python

'''
Simple "Square Detector" program.

Loads several images sequentially and tries to find squares in each image.
'''

# Python 2/3 compatibility
from __future__ import print_function
import sys

PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range

import math
import numpy as np 
import cv2

#计算三点的cos角
def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )
#计算图像中的手机屏幕区域
def find_squares(img):
    # 面积至少占五分之一以上
    width = img.shape[1]
    height = img.shape[0]
    area_min_limit = width * height / 5
    area_max_limit = width * height * 9 / 10
    img = cv2.GaussianBlur(img, (5, 5), 0)
    squares = []
    find_min_cos_value=0.3
    for gray in cv2.split(img):
        for thrs in xrange(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                bin = cv2.dilate(bin, None)
            else:
                _retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            contours, _hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
                countourArea=cv2.contourArea(cnt)
                if len(cnt) == 4 and  countourArea>area_min_limit and countourArea<area_max_limit  and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
                    if max_cos < find_min_cos_value:
                        find_min_cos_value=max_cos
                        squares.clear()
                        squares.append(cnt)
    return squares

#手机屏幕区域的四点定位
def divide_points_into_4_parts(line_nodes,   img_height,   img_width):
    left_top_line_nodes = []
    left_down_line_nodes = []
    right_top_line_nodes = []
    right_down_line_nodes = []
    res=[]
    height = img_height / 2
    width = img_width / 2
    for node in line_nodes[0]:
        if node[0]<width:
            if(node[1]<height):
                left_top_line_nodes.append(node[0])
                left_top_line_nodes.append(node[1])
            if(node[1]>height):
                left_down_line_nodes.append(node[0])
                left_down_line_nodes.append(node[1])
        if node[0] > width:
            if(node[1]<height):
                right_top_line_nodes.append(node[0])
                right_top_line_nodes.append(node[1])
            if(node[1]>height):
                right_down_line_nodes.append(node[0])
                right_down_line_nodes.append(node[1])


    res.append(left_top_line_nodes)
    res.append(left_down_line_nodes)
    res.append(right_top_line_nodes)
    res.append(right_down_line_nodes)
    return res

#根据四个定位点，做投影变换，得到有效图像
def perspective_transformation(final_points,img):

    leftTopX = final_points[0][0]
    leftTopY = final_points[0][1]
    leftDownX = final_points[1][0]
    leftDownY = final_points[1][1]
    rightTopX = final_points[2][0]
    rightTopY = final_points[2][1]
    rightDownX = final_points[3][0]
    rightDownY = final_points[3][1]
    newWidth = math.sqrt((leftTopX - rightTopX) * (leftTopX - rightTopX) + (leftTopY - rightTopY) * (leftTopY - rightTopY))
    newHeight = math.sqrt((leftTopX - leftDownX) * (leftTopX - leftDownX) + (leftTopY - leftDownY) * (leftTopY - leftDownY))
    # 定义对应的点
    points1 = np.float32(final_points)
    points2 = np.float32([[0,0],  [0,newHeight],[newWidth,0], [newWidth,newHeight]])
# 计算得到转换矩阵
    M = cv2.getPerspectiveTransform(points1, points2)
# 实现透视变换转换
    processed = cv2.warpPerspective(img,M,(int(newWidth), int(newHeight)))
    return processed
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'gif', 'GIF'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def getFinalImage(image):
    width = image.shape[1]
    height = image.shape[0]
    squares = find_squares(image)
    if len(squares) == 1:
        final_fps = divide_points_into_4_parts(squares, height, width)
        finalresult = perspective_transformation(final_fps, image)
    else:
        finalresult = image.copy()
    return  squares,finalresult

def main():
    global area_min_limit ,area_max_limit
    from glob import glob
    imgpath='ori_img/'
    #imgpath = 'from_other_camera/'
    for fn in glob(imgpath+'*.jpg'):
        if allowed_file(fn) == False:
            continue
        image = cv2.imread(fn)
        cv2.namedWindow('squares', 0)
        cv2.namedWindow('final_result', 0)
        width = image.shape[1]
        height = image.shape[0]

        FORCE_HEIGHT = 640
        rate = height / FORCE_HEIGHT

        cv2.resizeWindow('squares', int(width / rate), FORCE_HEIGHT)

        squares,finalresult=getFinalImage(image)

        cv2.resizeWindow('final_result', int(finalresult.shape[1] / (finalresult.shape[0] / FORCE_HEIGHT)), FORCE_HEIGHT)
        imageoutput = image.copy()
        if len(squares) == 1:
            for i in range(0, 4):
                cv2.putText(imageoutput, str(i) + str(squares[0][i]), tuple(squares[0][i]), cv2.FONT_HERSHEY_SIMPLEX, 2.5,
                            (0, 0, 255), 7)

        #cv2.imwrite('result.jpg',finalresult)
        cv2.drawContours( imageoutput, squares, -1, (0, 255, 0), 3 )
        cv2.imshow('squares', imageoutput)
        #cv2.imshow('final_result', imageoutput)
        cv2.imshow('final_result', finalresult)
        ch = cv2.waitKey()
        if ch == 27:
            break

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv2.destroyAllWindows()
