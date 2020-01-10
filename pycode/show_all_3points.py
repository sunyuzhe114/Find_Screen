import cv2
import numpy as np

blocksize = 128
blank_image = np.zeros([7 * blocksize, 12 * blocksize, 3], np.uint8) + 255
output_image = np.zeros([7 * blocksize, 12 * blocksize, 3], np.uint8) + 255
for i in range(0, 12):
    for j in range(0, 7):
        for m in range(0, 4):
            cv2.line(blank_image, pt1=(round(i * blocksize + blocksize / 4 * m), j * blocksize),
                     pt2=(round(i * blocksize + blocksize / 4 * m), j * blocksize + blocksize), color=(0, 255, 0))
            cv2.line(output_image, pt1=(round(i * blocksize + blocksize / 4 * m), j * blocksize),
                     pt2=(round(i * blocksize + blocksize / 4 * m), j * blocksize + blocksize), color=(0, 255, 0))
        for m in range(0, 4):
            cv2.line(blank_image, pt1=(i * blocksize, round(j * blocksize + blocksize / 4 * m)),
                     pt2=(i * blocksize + blocksize, round(j * blocksize + blocksize / 4 * m),), color=(0, 255, 0))
            cv2.line(output_image, pt1=(i * blocksize, round(j * blocksize + blocksize / 4 * m)),
                     pt2=(i * blocksize + blocksize, round(j * blocksize + blocksize / 4 * m),), color=(0, 255, 0))

pointOffset = [[round(blocksize / 4), round(blocksize / 4)],
               [round(blocksize / 2), round(blocksize / 4)],
               [round(blocksize * 3 / 4), round(blocksize / 4)],
               [round(blocksize / 4), round(blocksize / 2)],
               [round(blocksize / 2), round(blocksize / 2)],
               [round(blocksize * 3 / 4), round(blocksize / 2)],
               [round(blocksize / 4), round(blocksize * 3 / 4)],
               [round(blocksize / 2), round(blocksize * 3 / 4)],
               [round(blocksize * 3 / 4), round(blocksize * 3 / 4)]]
index = 0
point_list=[]
point_value_list=[]
for x in range(0, 7):
    for y in range(x + 1, 8):
        for z in range(y + 1, 9):
            value=int(1<<(8-x))+int(1<<(8-y))+int(1<<(8-z))
            #print("index",index ,'value=',value)
            point_value_list.append(value)
            point1_offset = pointOffset[x]
            point2_offset = pointOffset[y]
            point3_offset = pointOffset[z]
            i = index % 12
            j = int(index / 12)
            point_list.append((point1_offset,point2_offset,point3_offset))
            #cv2.putText(imageoutput, str(i) + str(squares[0][i]), tuple(squares[0][i]), cv2.FONT_HERSHEY_SIMPLEX, 2.5,
            #            (0, 0, 255), 7)

            cv2.circle(blank_image, (round(i * blocksize + point1_offset[0]), round(j * blocksize + +point1_offset[1])),
                       5, (0, 0, 255), thickness=-1)
            cv2.circle(blank_image, (round(i * blocksize + point2_offset[0]), round(j * blocksize + +point2_offset[1])),
                       5, (0, 0, 255), thickness=-1)
            cv2.circle(blank_image, (round(i * blocksize + point3_offset[0]), round(j * blocksize + +point3_offset[1])),
                       5, (0, 0, 255), thickness=-1)

            cv2.putText(blank_image,str(index), (round(i * blocksize  ), round(j * blocksize  )+18),cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                         (0,0 , 255),  1)
            index=index+1

for i in range(1, 12):
    cv2.line(blank_image, pt1=(round(i * blocksize), 0),
             pt2=(round(i * blocksize), blocksize * 7), color=(255, 0, 0))
    cv2.line(output_image, pt1=(round(i * blocksize), 0),
             pt2=(round(i * blocksize), blocksize * 7), color=(255, 0, 0))

for i in range(1, 7):
    cv2.line(blank_image, pt1=(0, round(i * blocksize)),
             pt2=(blocksize * 12, round(i * blocksize)), color=(255, 0, 0))
    cv2.line(output_image, pt1=(0, round(i * blocksize)),
             pt2=(blocksize * 12, round(i * blocksize)), color=(255, 0, 0))
#直接将序号写入后，就可以绘制圈圈
drawBlue=[6,10,11,12,17,21,22,23,24,26,27,31,38,43,47,51,52,53,55,58,61,62]
drawGreen=[5,8,14,18,19,20,32, 40, 41,42 ,46,48,49,50,56,57,64, 67,68,70,74, 78]

draw_digit_0_to_9=[6,10,11,12,17,21,22,23,24,26]
draw_alphabet_a_to_z=[27,31,38,43,47,51,52,53,55,58,61,62,5,8,14,18,19,20,32, 40, 41,42 ,46,48,49,50]
draw_remain=[56,57,64,67,68,70,74,78]

print("蓝色",len(drawBlue),"绿色",len(drawGreen))
print("数字",len(draw_digit_0_to_9),"字母",len(draw_alphabet_a_to_z),"预留",len(draw_remain))


for index in drawBlue:
    i = index % 12
    j = int(index / 12)
    cv2.circle(blank_image, (round(i * blocksize + pointOffset[4][0]), round(j * blocksize + +pointOffset[4][1])),
               63, (255, 0, 0), thickness=3)

for index in drawGreen:
    i = index % 12
    j = int(index / 12)
    cv2.circle(blank_image, (round(i * blocksize + pointOffset[4][0]), round(j * blocksize + +pointOffset[4][1])),
               63, (0, 255, 0), thickness=3)
#绘制数字字段
print('数字字段')
beginPos =0
for index in draw_digit_0_to_9:
    i = beginPos % 12
    j = int(beginPos / 12)
    point1_offset =point_list[index][0]
    point2_offset =point_list[index][1]
    point3_offset =point_list[index][2]
    cv2.circle(output_image, (round(i * blocksize + point1_offset[0]), round(j * blocksize +  point1_offset[1])),
               5, (0, 0, 255), thickness=-1)
    cv2.circle(output_image, (round(i * blocksize + point2_offset[0]), round(j * blocksize +  point2_offset[1])),
               5, (0, 0, 255), thickness=-1)
    cv2.circle(output_image, (round(i * blocksize + point3_offset[0]), round(j * blocksize +  point3_offset[1])),
               5, (0, 0, 255), thickness=-1)

    cv2.putText(output_image, str(beginPos)+'=>'+str(hex(point_value_list[index])), (round(i * blocksize), round(j * blocksize) + 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255), 2)
    print(str(beginPos)+'=>'+str(hex(point_value_list[index])))
    beginPos+=1

#绘制字母字段
print('字母字段')
beginPos =24
letter_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
for index in draw_alphabet_a_to_z:
    i = beginPos % 12
    j = int(beginPos / 12)
    point1_offset =point_list[index][0]
    point2_offset =point_list[index][1]
    point3_offset =point_list[index][2]
    cv2.circle(output_image, (round(i * blocksize + point1_offset[0]), round(j * blocksize   +point1_offset[1])),
               5, (0, 0, 255), thickness=-1)
    cv2.circle(output_image, (round(i * blocksize + point2_offset[0]), round(j * blocksize   +point2_offset[1])),
               5, (0, 0, 255), thickness=-1)
    cv2.circle(output_image, (round(i * blocksize + point3_offset[0]), round(j * blocksize   +point3_offset[1])),
               5, (0, 0, 255), thickness=-1)

    cv2.putText(output_image, letter_list[beginPos-24]+'=>'+str(hex(point_value_list[index])), (round(i * blocksize), round(j * blocksize) + 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0,0 ), 2)
    print(letter_list[beginPos-24]+'=>'+str(hex(point_value_list[index])))
    beginPos+=1

print('保留字段（待定）')
beginPos =72
letter_remain = ['A', 'B', 'C', 'D', 'E', 'F', ',', '.']
for index in draw_remain:
    i = beginPos % 12
    j = int(beginPos / 12)
    point1_offset =point_list[index][0]
    point2_offset =point_list[index][1]
    point3_offset =point_list[index][2]
    cv2.circle(output_image, (round(i * blocksize + point1_offset[0]), round(j * blocksize +  point1_offset[1])),
               5, (0, 0, 255), thickness=-1)
    cv2.circle(output_image, (round(i * blocksize + point2_offset[0]), round(j * blocksize +  point2_offset[1])),
               5, (0, 0, 255), thickness=-1)
    cv2.circle(output_image, (round(i * blocksize + point3_offset[0]), round(j * blocksize +  point3_offset[1])),
               5, (0, 0, 255), thickness=-1)

    cv2.putText(output_image, letter_remain[beginPos-72]+'=>'+str(hex(point_value_list[index])), (round(i * blocksize), round(j * blocksize) + 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0,0 ), 2)
    print(letter_remain[beginPos-72]+'=>'+str(hex(point_value_list[index])))
    beginPos+=1


print('起始帧')
beginPos =82
i = beginPos % 12
j = int(beginPos / 12)
stop_point_offset_list =[0,3,6,7,8]
value=0
for p in stop_point_offset_list:
    value += int(1 << (8 - p))
    cv2.circle(output_image, (round(i * blocksize + pointOffset[p][0]), round(j * blocksize   +pointOffset[p][1])),
               5, (0, 0, 255), thickness=-1)
cv2.putText(output_image, "begin", (round(i * blocksize), round(j * blocksize) + 18), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 0,0 ), 2)
print('起始帧'+'=>'+str(hex(value)))


print('中继帧')
beginPos =83
i = beginPos % 12
j = int(beginPos / 12)
#mid_point_offset_list =[1,3,4,5,7]
mid_point_offset_list =[0,1,2,5,8]
value=0
for p in mid_point_offset_list:
    value += int(1 << (8 - p))
    cv2.circle(output_image, (round(i * blocksize + pointOffset[p][0]), round(j * blocksize   +pointOffset[p][1])),
               5, (0, 0, 255), thickness=-1)
cv2.putText(output_image, "middle", (round(i * blocksize), round(j * blocksize) + 18), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 0,0 ), 2)
print('中继帧'+'=>'+str(hex(value)))
# for i in range(1, 5):
#     for j in range(1, 5):
# crossPoints.append((round(i * half_windowSize * 2 / 5), round(j * half_windowSize * 2 / 5)))
cv2.namedWindow("img", 0)
cv2.resizeWindow("img", 12 * blocksize, 7 * blocksize)
cv2.imshow("img", blank_image)
cv2.namedWindow("imgoutput", 0)
cv2.resizeWindow("imgoutput", 12 * blocksize, 7 * blocksize)
cv2.imshow("imgoutput", output_image)
cv2.waitKey(0)
