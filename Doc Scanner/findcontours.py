#findcontoursApril 06
import os
import numpy as np
import cv2
import pointtransform as pt

def draw_lines(image, lines):
    try:
        for line in lines:
            coords = line[0]
            cv2.line(image, (coords[0], coords[1]),
                     (coords[2], coords[3]),
                     [255, 255, 255], 3)
            
            
    except:
        pass
    
area = 5000
os.chdir('C:\\Users\\IBM_ADMIN\\Downloads')
img = cv2.imread('doc5.png',1)
img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('image',gray)
##cv2.imshow('image',img)
orig = img.copy()
filt = cv2.bilateralFilter(gray, 11, 17, 17)
##filt = cv2.bilateralFilter(img, 11, 17, 17)
##cv2.imshow('bilateral',filt)
##filt2=cv2.bilateralFilter(filt, 11, 17, 17)
##cv2.imshow('filt2',filt2)
edge1=cv2.Canny(filt,30, 200)
lines = cv2.HoughLinesP(edge1, 1, np.pi/180, 1, 100)
draw_lines(edge1, lines)
##for x1, y1, x2, y2 in lines[0]:
##    cv2.line(edge1, (x1,y1), (x2,y2), (255,255,255), 15)

##ope=cv2.morphologyEx(edge1.copy(), cv2.MORPH_OPEN, edge1)
##edge2=cv2.Canny(filt2, 30, 200)
cv2.imshow('edge1',edge1)
##cv2.imshow('edge2',edge2)
imc, contours, hierachy = cv2.findContours(edge1.copy(),
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
##contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
##for (i,c) in enumerate(contours):
##    M = cv2.moments(c)
##    cX=int(M["m10"]/M["m00"])
##    cY=int(M["m01"]/M["m00"])
##    cv2.putText(img, "#{}".format(i+1),(cX,cY),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,255),2)
##    cv2.drawContours(img, contours, -1, (0,255,0),1)
##    cv2.imshow('contours',img)
for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)


    if len(approx)==4:
        carea = cv2.contourArea(approx)
        print(carea)

        if carea>area:
            cv2.drawContours(img, approx, -1, (255, 0, 0), 7)
            cv2.imshow('contours', img)
            wrapped = pt.four_point_transform(img, approx.reshape(4, 2))
            wrapped = cv2.bilateralFilter(wrapped, 13, 17, 17)
            cv2.imshow('scanned', wrapped)
