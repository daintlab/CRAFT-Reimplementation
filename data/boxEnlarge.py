import numpy as np
import cv2
import math
from math import exp
import utils.config

def pointAngle(Apoint, Bpoint):
    angle = (Bpoint[1] - Apoint[1]) / ((Bpoint[0] - Apoint[0]) + 10e-8)
    return angle

def pointDistance(Apoint, Bpoint):
    return math.sqrt((Bpoint[1] - Apoint[1])**2 + (Bpoint[0] - Apoint[0])**2)

def lineBiasAndK(Apoint, Bpoint):

    K = pointAngle(Apoint, Bpoint)
    B = Apoint[1] - K*Apoint[0]
    return K, B

def getX(K, B, Ypoint):
    return int((Ypoint-B)/K)

def sidePoint(Apoint, Bpoint, h, w, placehold):

    K, B = lineBiasAndK(Apoint, Bpoint)
    angle = abs(math.atan(pointAngle(Apoint, Bpoint)))
    distance = pointDistance(Apoint, Bpoint)

    halfIncreaseDistance = utils.config.ENLARGEBOX_MAGINE * distance

    XaxisIncreaseDistance = abs(math.cos(angle) * halfIncreaseDistance)
    YaxisIncreaseDistance = abs(math.sin(angle) * halfIncreaseDistance)

    if placehold == 'leftTop':
        x1 = max(0, Apoint[0] - XaxisIncreaseDistance)
        y1 = max(0, Apoint[1] - YaxisIncreaseDistance)
    elif placehold == 'rightTop':
        x1 = min(w, Bpoint[0] + XaxisIncreaseDistance)
        y1 = max(0, Bpoint[1] - YaxisIncreaseDistance)
    elif placehold == 'rightBottom':
        x1 = min(w, Bpoint[0] + XaxisIncreaseDistance)
        y1 = min(h, Bpoint[1] + YaxisIncreaseDistance)
    elif placehold == 'leftBottom':
        x1 = max(0, Apoint[0] - XaxisIncreaseDistance)
        y1 = min(h, Apoint[1] + YaxisIncreaseDistance)

    return int(x1), int(y1)

def sidePoint_v1(Apoint, Bpoint, h, w, placehold):

    K, B = lineBiasAndK(Apoint, Bpoint)
    angle = abs(math.atan(pointAngle(Apoint, Bpoint)))
    distance = pointDistance(Apoint, Bpoint)

    XhalfIncreaseDistance = 0.75 * distance
    YhalfIncreaseDistance = 1.00 * distance

    XaxisIncreaseDistance = abs(math.cos(angle) * XhalfIncreaseDistance)
    YaxisIncreaseDistance = abs(math.sin(angle) * YhalfIncreaseDistance)

    if placehold == 'leftTop':
        x1 = max(0, Apoint[0] - XaxisIncreaseDistance)
        y1 = max(0, Apoint[1] - YaxisIncreaseDistance)
    elif placehold == 'rightTop':
        x1 = min(w, Bpoint[0] + XaxisIncreaseDistance)
        y1 = max(0, Bpoint[1] - YaxisIncreaseDistance)
    elif placehold == 'rightBottom':
        x1 = min(w, Bpoint[0] + XaxisIncreaseDistance)
        y1 = min(h, Bpoint[1] + YaxisIncreaseDistance)
    elif placehold == 'leftBottom':
        x1 = max(0, Apoint[0] - XaxisIncreaseDistance)
        y1 = min(h, Apoint[1] + YaxisIncreaseDistance)

    return int(x1), int(y1)

def sidePoint_v2(Apoint, Bpoint, h, w, placehold):

    K, B = lineBiasAndK(Apoint, Bpoint)
    angle = abs(math.atan(pointAngle(Apoint, Bpoint)))
    distance = pointDistance(Apoint, Bpoint)

    halfIncreaseDistance = 0.5 * distance

    XaxisIncreaseDistance = abs(math.cos(angle) * halfIncreaseDistance)
    YaxisIncreaseDistance = abs(math.sin(angle) * halfIncreaseDistance)

    if placehold == 'leftTop':
        if max(0, Apoint[0] - XaxisIncreaseDistance) == 0 or max(0, Apoint[1] - YaxisIncreaseDistance) == 0:
            # do not enlargeBox
            x1 = Apoint[0]
            y1 = Apoint[1]
        else:
            x1 = max(0, Apoint[0] - XaxisIncreaseDistance)
            y1 = max(0, Apoint[1] - YaxisIncreaseDistance)
    elif placehold == 'rightTop':
        if min(w, Bpoint[0] + XaxisIncreaseDistance) == w or max(0, Bpoint[1] - YaxisIncreaseDistance) == 0:
            # do not enlargeBox
            x1 = Bpoint[0]
            y1 = Bpoint[1]
        else:
            x1 = min(w, Bpoint[0] + XaxisIncreaseDistance)
            y1 = max(0, Bpoint[1] - YaxisIncreaseDistance)
    elif placehold == 'rightBottom':
        if min(w, Bpoint[0] + XaxisIncreaseDistance) == w or min(h, Bpoint[1] + YaxisIncreaseDistance) == h:
            # do not enlargeBox
            x1 = Bpoint[0]
            y1 = Bpoint[1]
        else:
            x1 = min(w, Bpoint[0] + XaxisIncreaseDistance)
            y1 = min(h, Bpoint[1] + YaxisIncreaseDistance)
    elif placehold == 'leftBottom':
        if max(0, Apoint[0] - XaxisIncreaseDistance) == 0 or min(h, Apoint[1] + YaxisIncreaseDistance) == h:
            # do not enlargeBox
            x1 = Apoint[0]
            y1 = Apoint[1]
        else:
            x1 = max(0, Apoint[0] - XaxisIncreaseDistance)
            y1 = min(h, Apoint[1] + YaxisIncreaseDistance)

    return int(x1), int(y1)

def sidePoint_v3(Apoint, Bpoint, h, w, placehold):

    K, B = lineBiasAndK(Apoint, Bpoint)
    angle = abs(math.atan(pointAngle(Apoint, Bpoint)))
    distance = pointDistance(Apoint, Bpoint)

    halfIncreaseDistance = 0.5 * distance

    XaxisIncreaseDistance = abs(math.cos(angle) * halfIncreaseDistance)
    YaxisIncreaseDistance = abs(math.sin(angle) * halfIncreaseDistance)

    if placehold == 'leftTop':
        if (Apoint[0] != 0 and max(0, Apoint[0] - XaxisIncreaseDistance) == 0) or (Apoint[1] != 0 and max(0, Apoint[1] - YaxisIncreaseDistance) == 0):
            # do not enlargeBox
            x1 = Apoint[0]
            y1 = Apoint[1]
        else:
            x1 = max(0, Apoint[0] - XaxisIncreaseDistance)
            y1 = max(0, Apoint[1] - YaxisIncreaseDistance)
    elif placehold == 'rightTop':
        if (Bpoint[0] != w and min(w, Bpoint[0] + XaxisIncreaseDistance) == w) or (Bpoint[1] != 0 and max(0, Bpoint[1] - YaxisIncreaseDistance) == 0):
            # do not enlargeBox
            x1 = Bpoint[0]
            y1 = Bpoint[1]
        else:
            x1 = min(w, Bpoint[0] + XaxisIncreaseDistance)
            y1 = max(0, Bpoint[1] - YaxisIncreaseDistance)
    elif placehold == 'rightBottom':
        if (Bpoint[0] != w and min(w, Bpoint[0] + XaxisIncreaseDistance) == w) or (Bpoint[1] != h and min(h, Bpoint[1] + YaxisIncreaseDistance) == h):
            # do not enlargeBox
            x1 = Bpoint[0]
            y1 = Bpoint[1]
        else:
            x1 = min(w, Bpoint[0] + XaxisIncreaseDistance)
            y1 = min(h, Bpoint[1] + YaxisIncreaseDistance)
    elif placehold == 'leftBottom':
        if (Apoint[0] != 0 and max(0, Apoint[0] - XaxisIncreaseDistance) == 0) or (Apoint[1] != h and min(h, Apoint[1] + YaxisIncreaseDistance) == h):
            # do not enlargeBox
            x1 = Apoint[0]
            y1 = Apoint[1]
        else:
            x1 = max(0, Apoint[0] - XaxisIncreaseDistance)
            y1 = min(h, Apoint[1] + YaxisIncreaseDistance)

    return int(x1), int(y1)

def sidePoint_v4(Apoint, Bpoint, h, w, placehold):

    K, B = lineBiasAndK(Apoint, Bpoint)
    angle = abs(math.atan(pointAngle(Apoint, Bpoint)))
    distance = pointDistance(Apoint, Bpoint)

    XhalfIncreaseDistance = 0.5 * distance
    YhalfIncreaseDistance = 0.75 * distance

    XaxisIncreaseDistance = abs(math.cos(angle) * XhalfIncreaseDistance)
    YaxisIncreaseDistance = abs(math.sin(angle) * YhalfIncreaseDistance)

    if placehold == 'leftTop':
        if (Apoint[0] != 0 and max(0, Apoint[0] - XaxisIncreaseDistance) == 0) or (Apoint[1] != 0 and max(0, Apoint[1] - YaxisIncreaseDistance) == 0):
            # do not enlargeBox
            x1 = Apoint[0]
            y1 = Apoint[1]
        else:
            x1 = max(0, Apoint[0] - XaxisIncreaseDistance)
            y1 = max(0, Apoint[1] - YaxisIncreaseDistance)
    elif placehold == 'rightTop':
        if (Bpoint[0] != w and min(w, Bpoint[0] + XaxisIncreaseDistance) == w) or (Bpoint[1] != 0 and max(0, Bpoint[1] - YaxisIncreaseDistance) == 0):
            # do not enlargeBox
            x1 = Bpoint[0]
            y1 = Bpoint[1]
        else:
            x1 = min(w, Bpoint[0] + XaxisIncreaseDistance)
            y1 = max(0, Bpoint[1] - YaxisIncreaseDistance)
    elif placehold == 'rightBottom':
        if (Bpoint[0] != w and min(w, Bpoint[0] + XaxisIncreaseDistance) == w) or (Bpoint[1] != h and min(h, Bpoint[1] + YaxisIncreaseDistance) == h):
            # do not enlargeBox
            x1 = Bpoint[0]
            y1 = Bpoint[1]
        else:
            x1 = min(w, Bpoint[0] + XaxisIncreaseDistance)
            y1 = min(h, Bpoint[1] + YaxisIncreaseDistance)
    elif placehold == 'leftBottom':
        if (Apoint[0] != 0 and max(0, Apoint[0] - XaxisIncreaseDistance) == 0) or (Apoint[1] != h and min(h, Apoint[1] + YaxisIncreaseDistance) == h):
            # do not enlargeBox
            x1 = Apoint[0]
            y1 = Apoint[1]
        else:
            x1 = max(0, Apoint[0] - XaxisIncreaseDistance)
            y1 = min(h, Apoint[1] + YaxisIncreaseDistance)

    return int(x1), int(y1)

# not completed
def sidePoint_v5(Apoint, Bpoint, h, w, placehold):

    K, B = lineBiasAndK(Apoint, Bpoint)
    angle = abs(math.atan(pointAngle(Apoint, Bpoint)))
    distance = pointDistance(Apoint, Bpoint)

    halfIncreaseDistance = 0.5 * distance

    XaxisIncreaseDistance = abs(math.cos(angle) * halfIncreaseDistance)
    YaxisIncreaseDistance = abs(math.sin(angle) * halfIncreaseDistance)

    if placehold == 'leftTop':
        # x축, y축 경계 모두 넘을 경우 => 확대하지 않음
        if (Apoint[0] != 0 and max(0, Apoint[0] - XaxisIncreaseDistance) == 0) and (Apoint[1] != 0 and max(0, Apoint[1] - YaxisIncreaseDistance) == 0):
            x1 = Apoint[0]
            y1 = Apoint[1]
        # x축만 경계 넘을 경우 => y축만 확대
        elif (Apoint[0] != 0 and max(0, Apoint[0] - XaxisIncreaseDistance) == 0) and not (Apoint[1] != 0 and max(0, Apoint[1] - YaxisIncreaseDistance) == 0):
            x1 = Apoint[0]
            y1 = max(0, Apoint[1] - YaxisIncreaseDistance)
        # y축만 경계 넘을 경우 => x축만 확대
        elif not (Apoint[0] != 0 and max(0, Apoint[0] - XaxisIncreaseDistance) == 0) and (Apoint[1] != 0 and max(0, Apoint[1] - YaxisIncreaseDistance) == 0):
            x1 = max(0, Apoint[0] - XaxisIncreaseDistance)
            y1 = Apoint[1]
        # 모두 경계 넘지 않을 경우 => 확대
        else:
            x1 = max(0, Apoint[0] - XaxisIncreaseDistance)
            y1 = max(0, Apoint[1] - YaxisIncreaseDistance)
    elif placehold == 'rightTop':
        if (Bpoint[0] != w and min(w, Bpoint[0] + XaxisIncreaseDistance) == w) and (Bpoint[1] != 0 and max(0, Bpoint[1] - YaxisIncreaseDistance) == 0):
            x1 = Bpoint[0]
            y1 = Bpoint[1]
        else:
            x1 = min(w, Bpoint[0] + XaxisIncreaseDistance)
            y1 = max(0, Bpoint[1] - YaxisIncreaseDistance)
    elif placehold == 'rightBottom':
        if (Bpoint[0] != w and min(w, Bpoint[0] + XaxisIncreaseDistance) == w) or (Bpoint[1] != h and min(h, Bpoint[1] + YaxisIncreaseDistance) == h):
            # do not enlargeBox
            x1 = Bpoint[0]
            y1 = Bpoint[1]
        else:
            x1 = min(w, Bpoint[0] + XaxisIncreaseDistance)
            y1 = min(h, Bpoint[1] + YaxisIncreaseDistance)
    elif placehold == 'leftBottom':
        if (Apoint[0] != 0 and max(0, Apoint[0] - XaxisIncreaseDistance) == 0) or (Apoint[1] != h and min(h, Apoint[1] + YaxisIncreaseDistance) == h):
            # do not enlargeBox
            x1 = Apoint[0]
            y1 = Apoint[1]
        else:
            x1 = max(0, Apoint[0] - XaxisIncreaseDistance)
            y1 = min(h, Apoint[1] + YaxisIncreaseDistance)

    return int(x1), int(y1)

# 将box扩大1.5倍
def enlargebox(box, h, w):


    # box = [Apoint, Bpoint, Cpoint, Dpoint]
    Apoint, Bpoint, Cpoint, Dpoint = box
    K1, B1 = lineBiasAndK(box[0], box[2])
    K2, B2 = lineBiasAndK(box[3], box[1])
    X = (B2 - B1)/(K1 - K2)
    Y = K1 * X + B1
    center = [X, Y]

    x1, y1 = sidePoint_v1(Apoint, center, h, w, 'leftTop')
    x2, y2 = sidePoint_v1(center, Bpoint, h, w, 'rightTop')
    x3, y3 = sidePoint_v1(center, Cpoint, h, w, 'rightBottom')
    x4, y4 = sidePoint_v1(Dpoint, center, h, w, 'leftBottom')
    newcharbox = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    return newcharbox
