import numpy as np
import cv2
import time

acceptedFGThreshold = 150


# define a video capture object
cap = cv2.VideoCapture(0)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
  
# initializing subtractor 
fgbg = cv2.createBackgroundSubtractorMOG2()

fgbg.setBackgroundRatio(0.8)
fgbg.setComplexityReductionThreshold(0.8)
fgbg.setHistory(20)


while(True):
	
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)

    fgmask = cv2.erode(fgmask, kernel, cv2.BORDER_REFLECT)
    # Background area using Dialation
    fgmask = cv2.dilate(fgmask, kernel, iterations = 1)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations = 2)

    ret, fgmask = cv2.threshold(fgmask, 110, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    

    totalX = 0
    totalY = 0

    numberOfXY = 0    

    for cnt in contours :
        
        if cv2.arcLength(cnt, True) < 200:# and cv2.arcLength(cnt, True) > 20:
            continue
        approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)

        #image = cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2) 

        n = approx.ravel() 
        i = 0
    
        for j in n :
            if(i % 2 == 0):
                totalX += n[i]
                totalY += n[i + 1]
                numberOfXY += 1
            i += 1
    if numberOfXY > 0:
        avgX = totalX / numberOfXY
        avgY = totalY / numberOfXY

    print(avgX, avgY)
    if (avgX != 0 and avgY != 0):
        image = cv2.circle(frame, (int(avgX), int(avgY)), 20, (255,0,0), 3)

    #avg_color_per_row = np.average(fgmask, axis=0)
    #avg_color = np.average(avg_color_per_row, axis=0)
    #print(avg_color)
    #if avg_color > acceptedFGThreshold:
        #fgbg.setHistory(5)
    #else:
        #fgbg.setHistory(20)
        

    cv2.imshow('frame', image)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
  
cap.release()
cv2.destroyAllWindows()
