import numpy as np
import cv2
import time

powerUsageThrotle = 0 # sleep in s to reduce power usage 

peopleInFrame = 0
avgX1 = 0
avgY1 = 0

avgX2 = 0
avgY2 = 0

maxX = 0
minX = 0

avgDistThreshold = 0.3
minMaxDistThreshold = 0.8
acceptedFGThreshold = 90

cap = cv2.VideoCapture(0)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
fgbg = cv2.createBackgroundSubtractorMOG2()

fgbg.setBackgroundRatio(0.8)
fgbg.setComplexityReductionThreshold(0.8)
fgbg.setHistory(30)


while(True):
	
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # recording and detecting fg
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)

    height = frame.shape[0] 
    width = frame.shape[1]

    # cleaning the image
    fgmask = cv2.erode(fgmask, kernel, cv2.BORDER_REFLECT)
    fgmask = cv2.dilate(fgmask, kernel, iterations = 1)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations = 2)

    # finding and processing contours
    ret, fgmask = cv2.threshold(fgmask, 110, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    totalX1 = 0
    totalY1 = 0

    totalX2 = 0
    totalY2 = 0

    numberOfXY1 = 0    
    numberOfXY2 = 0    

    image = frame
    for cnt in contours:
        
        if cv2.arcLength(cnt, True) < 200:# and cv2.arcLength(cnt, True) > 20:
            continue

        avg_color_per_row = np.average(fgmask, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        print(avg_color)
        if avg_color > acceptedFGThreshold:
            break

        approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)

        image = cv2.drawContours(image, [approx], -1, (0, 255, 0), 2) 

        n = approx.ravel() 
        i = 0
    
        for j in n :
            if(i % 2 == 0):
                try:
                    if n[i] > (avgX1 + avgX2) / 2: # if the contours X pos is on the right side
                        totalX1 += n[i]
                        totalY1 += n[i + 1]
                        numberOfXY1 += 1
                    else:                   # if it is on the left side
                        totalX2 += n[i]
                        totalY2 += n[i + 1]
                        numberOfXY2 += 1
                except:                     # also the left side
                    totalX2 += n[i]
                    totalY2 += n[i + 1]
                    numberOfXY2 += 1
                
                # getting min and max of the X
                if minX == 0: 
                    minX = n[i]
                else:
                    minX = min(n[i], minX)

                maxX = max(maxX, n[i])
            i += 1

    # Getting avarages
    if numberOfXY1 > 0:
        avgX1 = totalX1 / numberOfXY1
        avgY1 = totalY1 / numberOfXY1
    
    if numberOfXY2 > 0:
        avgX2 = totalX2 / numberOfXY2
        avgY2 = totalY2 / numberOfXY2

    if avgX1 - avgX2 > width * avgDistThreshold and maxX - minX > width * minMaxDistThreshold: # if the avarage contours are separated enough
        image = cv2.circle(image, (int(avgX1), int(avgY1)), 20, (255,0,0), 3)
        image = cv2.circle(image, (int(avgX2), int(avgY2)), 20, (255,0,0), 3)
        peopleInFrame = 2

    elif avgX1 + avgX2 != 0 and avgY1 + avgY2 != 0 and numberOfXY1 + numberOfXY2 > 0: # if there are contours
        image = cv2.circle(image, (int((avgX1 + avgX2) / (2)), int((avgY1 + avgY2) / (2))), 20, (255,0,0), 3)
        peopleInFrame = 1

    else: # no contours
        peopleInFrame = 0

        

    cv2.imshow('frame', image)
    time.sleep(powerUsageThrotle)
  
cap.release()
cv2.destroyAllWindows()
