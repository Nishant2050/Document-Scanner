import numpy as np
import cv2

def order_points(pts):
    rect=np.zeros((4, 2),dtype="float32")
    s=np.sum(pts, axis=1)
    rect[0]= pts[np.argmin(s)]
    rect[2]=pts[np.argmax(s)]
    d=np.diff(pts, axis=1)
    rect[1]=pts[np.argmin(d)]
    rect[3]=pts[np.argmax(d)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA=np.sqrt(((tl[0]-tr[0])**2)+((tl[1]-tr[1])**2))
    widthB=np.sqrt(((bl[0]-br[0])**2)+((bl[1]-br[1])**2))
    maxwidth=max(int(widthA), int(widthB))

    heightA=np.sqrt(((tl[0]-bl[0])**2)+((tl[1]-bl[1])**2))
    heightB=np.sqrt(((tr[0]-br[0])**2)+((tr[1]-br[1])**2))
    maxheight=max(int(heightA), int(heightB))

    dist=np.array([[0, 0],
                  [maxwidth, 0],
                  [maxwidth, maxheight],
                  [0, maxheight]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dist)
    wrapped = cv2.warpPerspective(image, M, (maxwidth, maxheight))

    return wrapped

    
