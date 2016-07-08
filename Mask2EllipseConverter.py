import cv2
import numpy as np

class Mask2EllipseConverter(object):
    '''
    Description:
    Converts mask to ellipse.
    
    Args:
    mask - mask in opencv's Mat format. Must have one channel. Pixels values must be or 0 (for background) either 1 (for foreground).
    
    Returns:
    ellipse - an opencv RotatedRect, ellipse[0] - center point, ellipse[1] - size (width and height), ellipse[2] - angle
    '''
    def convert(self, mask):
        self._checkMaskCorrectness(mask)

        copy = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        img, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            raise ValueError("Given mask does not contain any object")
        
        ellipse = cv2.fitEllipse(contours[0])
        
        return ellipse
        

    def _checkMaskCorrectness(self, mask):
        if len(mask.shape) != 2:
            raise ValueError("Given mask must be in grayscale format.")
        
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(mask)
        if minVal != 0 and maxVal != 1:
            raise ValueError("Pixels for object must equal to 1, pixels for background must equal to 0.")