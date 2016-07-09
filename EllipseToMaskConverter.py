import cv2
import numpy as np

class EllipseToMaskConverter(object):
    '''
    Description:
    Converts ellipse contour points to binary mask matrix
    
    Args:
    contours - list of points in format [[x1, y1], [x2, y2], ..., [xn, ym]]. Points must be in correct order! contours[i] point should be a previous point to contours[i+1]
    maskWidth - width of output mask
    maskHeight - height of output mask
    
    Returns:
    mask - binary mask with values: 0 for background and 1 for foreground
    '''
    def convertFromContourPoints(self, contours, maskWidth, maskHeight):
        self._checkContoursCorrectness(contours)

        contoursTmp = []
        for i in range(0, len(contours)):
            contoursTmp.append([contours[i]])
        contours = np.array([contoursTmp])

        mask = np.zeros((maskHeight, maskWidth), dtype=np.uint8)
    
        mask = cv2.fillPoly(mask, contours, 1, 8)

        return mask


    '''
    Description:
    Converts rotated rectangle object to binary mask matrix
    
    Args:
    rotatedRectangle - instance of opencv's rotatedRect class
    maskWidth - width of output mask
    maskHeight - height of output mask
    
    Returns:
    mask - binary mask with values: 0 for background and 1 for foreground
    '''
    def convertFromRotatedRectangle(self, rotatedRect, maskWidth, maskHeight):
        self._checkRotatedRectCorrectness(rotatedRect)

        mask = np.zeros((maskHeight, maskWidth), dtype=np.uint8)   
        mask = cv2.ellipse(mask, rotatedRect, 1, -1)
        return mask
        

    def _checkContoursCorrectness(self, contours):
        if len(contours) < 3:
            raise ValueError("List of contours must contain at least 3 points.")
        
        for i in range(0, len(contours)):
            if len(contours[i]) != 2:
                raise ValueError("Point no. " + i + " is not a 2D points.")


    def _checkRotatedRectCorrectness(self, rotatedRect):
        if len(rotatedRect) != 3:
            raise ValueError("Incorrect rotated rect object.")
        if len(rotatedRect[0]) != 2:
            raise ValueError("Center point is not a 2D point.")
        if len(rotatedRect[1]) != 2:
            raise ValueError("Theres more dimensions than width and height only.")
        if not np.isscalar(rotatedRect[2]):
            raise ValueError("Rotating angle should be a scalar")