import cv2

class Mask2RunLengthConverter(object):
    
    '''
    Description:
    Converts mask to run length format in order down-then-right.
    
    Args:
    mask - mask in opencv's Mat format. Must have one channel. Pixels values must be or 0 (for background) either 1 (for foreground).
    
    Returns:
    runLength - 1D list of numbers in run length format.
    '''
    def convert(self, mask):
        self._checkMaskCorrectness(mask)

        runLength = []        
        onRun = False
        onRunCounter = 0
        for x in xrange(mask.shape[1]):
            for y in xrange(mask.shape[0]):
                if mask[y, x] == 1 and not onRun:
                    onRun = True
                    runLength.append((y+1) + x*mask.shape[1])
                    onRunCounter += 1
                elif mask[y, x] == 1 and onRun:
                    onRunCounter += 1
                elif mask[y, x] == 0 and onRun:
                    runLength.append(onRunCounter)
                    onRunCounter = 0
                    onRun = False
                
        if onRun:
            runLength.append(onRunCounter)

        return runLength

    def _checkMaskCorrectness(self, mask):
        if len(mask.shape) != 2:
            raise ValueError("Given mask must be in grayscale format.")
        
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(mask)
        if minVal != 0 and maxVal != 1:
            raise ValueError("Pixels for object must equal to 1, pixels for background must equal to 0.")