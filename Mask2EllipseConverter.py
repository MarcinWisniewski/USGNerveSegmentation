import cv2
import numpy as np

class Mask2EllipseConverter(object):
    '''
    Description:
    Converts mask to ellipse.
    
    Args:
    mask - mask in opencv's Mat format. Must have one channel. Pixels values must be or 0 (for background) either 1 (for foreground).
    
    Returns:
    contours - numpy array of points on edge foreground - background in order (x,y) - column, then row index
    ellispePoints - numpy array of points on egde of fitted ellipse in order (x,y) - column, then row index
    rotatedRect - an opencv RotatedRect, ellipse[0] - center point, ellipse[1] - size (width and height), ellipse[2] - angle
    '''
    def convert(self, mask):
        self._checkMaskCorrectness(mask)

        copy = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        img, contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        if len(contours) == 0:
            return [None, None, [(0, 0), (0, 0), 0.0]]

        rotatedRect = cv2.fitEllipse(contours[0])
        
        cpY, cpX = np.array(rotatedRect[0], int)
        height, width = np.array(rotatedRect[1], int)
        angle = (int)(rotatedRect[2])

        ellispePoints = cv2.ellipse2Poly((cpY, cpX), (height//2, width//2), angle, 0, 360, 1)

        contoursTmp = []
        for i in range(0, len(contours[0])):
            px = contours[0][i][0][0]
            py = contours[0][i][0][1]
            contoursTmp.append([px, py])
        contours = np.array(contoursTmp)

        return contours, ellispePoints, rotatedRect

    def _checkMaskCorrectness(self, mask):
        if len(mask.shape) != 2:
            raise ValueError("Given mask must be in grayscale format.")
        
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(mask)
        if minVal != 0 and maxVal != 1:
            raise ValueError("Pixels for object must equal to 1, pixels for background must equal to 0.")