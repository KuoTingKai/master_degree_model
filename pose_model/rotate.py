import numpy as np
import cv2
import csv
import os
def gray_img(img):

    B = (img / 256).astype(np.uint8)
    R = (img % 256).astype(np.uint8)
    G = np.zeros((424, 512)).astype(np.uint8)

    a = cv2.merge((B,G,R))
# for x in range(len(img)):
    #     for y in range(len(img[0])):
    #         if img[x][y] < 700 or img[x][y] > 1000:
    #             img[x][y] = 0
    #         else:
    #             img[x][y] = img[x][y] - 700
    return a     



def relative_feature(img,GT,box):
    # feature = []
    delta = []
    x = int(float(GT[0]))
    y = int(float(GT[1]))

    depth = img[y][x]
    for offset_y in range(-box,box):
        # print(offset_y)
        for offset_x in range(-box,box):
            # if offset_y == 0 and offset_x == 0: continue
            # delta_1 = full_image[164 + y + offset_y][169 + x + offset_x] - depth
            delta_1 = img[y + offset_y][x + offset_x] - depth
            delta.append(delta_1)
    # for i in range(-box,box):
    #     if i < 0:
    #         delta_2 = img[y + i][x + i] - img[y - i][x - i]
    #     elif i == 0:
    #         continue
    #     else:
    #         delta_2 = img[y - i][x - i] - img[y + i][x + i]
    #     # delta_2 = delta[keypoint * abs(box * box * 4) + i * box + i] - delta[keypoint * abs(box * box * 4) + (i + 1) * box - i - 1]
    #     delta.append(delta_2)

    return delta

def rotate(ps,m):
    pts = np.float32(ps).reshape([-1, 2])  # 要映射的点
    pts = np.hstack([pts, np.ones([len(pts), 1])]).T
    target_point = np.dot(m, pts)
    target_point = [(target_point[0][x],target_point[1][x]) for x in range(len(target_point[0]))]
    target_point = np.array(target_point)
    
    return target_point[0]


def head_rotate(depth_img,GT,box):
    # img = cv2.imread(img_name)
    rotated_img = depth_img
    (h, w) = depth_img.shape[:2]
    center = (w / 2,h / 2)
    angle = 0
    rotated_GT = GT
    feature = []
    max_angle = False
    a = (GT[1],GT[2])
    b = (GT[4],GT[5])
    c = (GT[7],GT[8])
    d = (GT[10],GT[11])
    e = (GT[13],GT[14])
    f = (GT[16],GT[17])
    while((a[1] > min(b[1],c[1],d[1],e[1],f[1])) 
     or (b[0] < c[0])
     or (int(float(b[1])) - int(float(c[1]))) > 2):
        angle = angle + 1
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_img = cv2.warpAffine(depth_img, M, (w, h))
        a = np.dot(M,(int(float(rotated_GT[1])),int(float(rotated_GT[2])),1))
        b = np.dot(M,(int(float(rotated_GT[4])),int(float(rotated_GT[5])),1))
        c = np.dot(M,(int(float(rotated_GT[7])),int(float(rotated_GT[8])),1))
        d = np.dot(M,(int(float(rotated_GT[10])),int(float(rotated_GT[11])),1))
        e = np.dot(M,(int(float(rotated_GT[13])),int(float(rotated_GT[14])),1))
        f = np.dot(M,(int(float(rotated_GT[16])),int(float(rotated_GT[17])),1))

        if (angle >= 360):
            break

    if (angle == 360):
        max_angle = True
        return feature,max_angle
    
    feature.extend(relative_feature(rotated_img,a,box))
    feature.extend(relative_feature(rotated_img,b,box))
    feature.extend(relative_feature(rotated_img,c,box))
    feature.extend(relative_feature(rotated_img,d,box))
    return feature, max_angle

def mid_body_rotate(depth_img,GT,box):
    # img = cv2.imread(img_name)
    rotated_img = depth_img
    (h, w) = depth_img.shape[:2]
    center = (w / 2,h / 2)
    angle = 0
    rotated_GT = GT
    feature = []
    max_angle = False
    a = (GT[1],GT[2])
    b = (GT[4],GT[5])
    c = (GT[7],GT[8])
    d = (GT[10],GT[11])
    e = (GT[13],GT[14])
    f = (GT[16],GT[17])
    while( abs(int(float(d[0])) - int(float(e[0])) > 2 ) or
    (a[1] > min(b[1],c[1],d[1],e[1],f[1]))):
        angle = angle + 1
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_img = cv2.warpAffine(depth_img, M, (w, h))
        a = np.dot(M,(int(float(rotated_GT[1])),int(float(rotated_GT[2])),1))
        b = np.dot(M,(int(float(rotated_GT[4])),int(float(rotated_GT[5])),1))
        c = np.dot(M,(int(float(rotated_GT[7])),int(float(rotated_GT[8])),1))
        d = np.dot(M,(int(float(rotated_GT[10])),int(float(rotated_GT[11])),1))
        e = np.dot(M,(int(float(rotated_GT[13])),int(float(rotated_GT[14])),1))
        f = np.dot(M,(int(float(rotated_GT[16])),int(float(rotated_GT[17])),1))
        
        if (angle >= 360):
            # print('aaaaaaa')
            break
    if (angle == 360):
        max_angle = True
        return feature,max_angle
    # rotated_img = gray_img(rotated_img)
    # cv2.rectangle(rotated_img,(int(float(e[0]))-box,int(float(e[1]))-box),(int(float(e[0]))+box,int(float(e[1]))+box),(0,255,0),2)
    # cv2.imshow("a",rotated_img)
    # cv2.waitKey(33)
    feature.extend(relative_feature(rotated_img,e,box))

    return feature, max_angle

def body_tail_rotate(depth_img,GT,box):
    # img = cv2.imread(depth_img)
    rotated_img = depth_img
    (h, w) = depth_img.shape[:2]
    center = (w / 2,h / 2)
    angle = 0
    rotated_GT = GT
    feature = []
    max_angle = False
    a = (GT[1],GT[2])
    b = (GT[4],GT[5])
    c = (GT[7],GT[8])
    d = (GT[10],GT[11])
    e = (GT[13],GT[14])
    f = (GT[16],GT[17])
    while( abs(int(float(e[0])) - int(float(f[0]))) > 2 or 
    (a[1] > min(b[1],c[1],d[1],e[1],f[1])) or
    (d[1] > min(e[1],f[1])) or
    (e[1] > f[1])):
        angle = angle + 1
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_img = cv2.warpAffine(depth_img, M, (w, h))
        # a = rotate((GT[1],GT[2]),M)
        # b = rotate((GT[4],GT[5]),M)
        # c = rotate((GT[7],GT[8]),M)
        # d = rotate((GT[10],GT[11]),M)
        # e = rotate((GT[13],GT[14]),M)
        # f = rotate((GT[16],GT[17]),M)
        a = np.dot(M,(int(float(GT[1])),int(float(GT[2])),1))
        b = np.dot(M,(int(float(GT[4])),int(float(GT[5])),1))
        c = np.dot(M,(int(float(GT[7])),int(float(GT[8])),1))
        d = np.dot(M,(int(float(GT[10])),int(float(GT[11])),1))
        e = np.dot(M,(int(float(GT[13])),int(float(GT[14])),1))
        f = np.dot(M,(int(float(GT[16])),int(float(GT[17])),1))

        if (angle >= 360):
            # print('aaaaaaa')
            break
    if (angle == 360):
        max_angle = True
        # feature.extend(relative_feature(rotated_img,f,box))
        return feature,max_angle
    # rotated_img = gray_img(rotated_img)
    # depth_img = gray_img(depth_img)
    # cv2.imshow('b',depth_img)
    # # cv2.waitKey(0)
    # cv2.rectangle(rotated_img,(int(float(e[0]))-box,int(float(e[1]))-box),(int(float(e[0]))+box,int(float(e[1]))+box),(0,255,0),2)
    # cv2.imshow("a",rotated_img)
    # cv2.waitKey(33)
    feature.extend(relative_feature(rotated_img,f,box))
    return feature, max_angle

# if __name__ == '__main__':
#     # img = cv2.imread('F:/choice_depth_image/suffle6/predrug_DCW_OK/frame1.png')
#     # cv2.imshow('ori',img)
#     # (h, w) = img.shape[:2]
#     # center = (w / 2,h / 2)
#     # M = cv2.getRotationMatrix2D(center, 90, 1.0)
#     # print(M)
#     # rotated = cv2.warpAffine(img, M, (w, h))
#     # # print()
#     # print(np.dot(M,(163,241,1)))
#     # cv2.imwrite('C:/Users/kevin/Desktop/a.png',rotated)
#     # cv2.waitKey(0)
#     name = 'DS'
#     treatment_csv = 'F:/kinect/csv/treatment_' + name + '.csv'
#     treatment_img_folder = 'F:/choice_depth_image/suffle6/treatment_' + name + '_OK'
#     treatment_GT = []

#     count = 0
#     with open(treatment_csv, newline='') as csvfile:
#         rows = csv.reader(csvfile)
#         for row in rows:
#             count = count + 1
#             if count > 3:
#                 treatment_GT.append(row)
#                 # print(treatment_GT)
#             else:
#                 continue 

#     treatment_img_list = [_ for _ in os.listdir(treatment_img_folder) if _.endswith('.png')]

#     treatment_img_list.sort(key=lambda x:int(x[5:-4]))

#     box = 5
#     count = 0
#     for i in range(len(treatment_img_list)):
#         treatment_name = os.path.join(treatment_img_folder,treatment_img_list[i])
#         feature = head_rotate(treatment_name,treatment_GT[i],box)
#         # rotate_img, rotate_GT, angle = head_rotate(treatment_name,treatment_GT[i],box)
#         cv2.imshow('a',rotate_img)
#         cv2.waitKey(33)
#         if angle == 360:
#             count = count + 1
#         print(i)
#     print(count)
    
    
    # print(rotate_GT)


