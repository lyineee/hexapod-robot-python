import cv2 
import numpy as np 

def get_px_num(img):
    left,right=0,0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if i<img.shape[0]/2:
                left+=img[i][j]
            else:
                right+=img[i][j]
    return left,right


def img_show(img):
    cv2.destroyAllWindows()
    cv2.namedWindow('test')
    cv2.imshow('test',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def binarization(img,th=128):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j]>th:
                img[i][j]=1
            else:
                img[i][j]=0
    return img


def edge_det(img):
    # 转灰度图片
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=img.astype(np.uint8)
    #binarization
    ret, binary = cv2.threshold(img,128,255,cv2.THRESH_BINARY)
    binary=cv2.bitwise_not(binary)

    #filter
    # 1
    # binary = cv2.medianBlur(binary, 7)
    # result=cv2.Canny(binary ,3, 6,20)

    # 2
    try:
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        result=cv2.drawContours(np.zeros((img.shape[0],img.shape[1]))+255,c, -1, (0,0,0), 3)
    except:
        result=0
    return result