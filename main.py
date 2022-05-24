import numpy as np, cv2
import wave, pyaudio, audioop
import time

import pyaudio
import wave
import audioop

def talk(numberOfPeople, volume):
    if numberOfPeople == 1:
        print("Talk to one person!", volume)
        return True

    elif numberOfPeople == 2:
        print("Talk to multiple people!", volume)
        return True

    else:
        print("This elevator is empty!", volume)
        return False

# program params
powerUsageThrotle = 0 # sleep in s to reduce power usage 

people2Remember = 10
rememebrdPeople = [0] * people2Remember

audio2Remember = 10
rememberdAudio = [0] * audio2Remember

audioThreshold = 2000

talkDelay = 4
lastTalkTime = 0
frameTimer = 0

state = "tyst"

# visual params
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

# audio params
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100


pyAud = pyaudio.PyAudio()

""""
device_count = pyAud.get_device_count()
for i in range(0, device_count):
        print("Name: " + str(pyAud.get_device_info_by_index(i)["name"]))
        print("Index: " + str(pyAud.get_device_info_by_index(i)["index"]))
        print("\n")
"""
stream = pyAud.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

while(True):
    startTime = time.time()

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

        avgRowColor = np.average(fgmask, axis=0)
        avgColor = np.average(avgRowColor, axis=0)
        if avgColor > acceptedFGThreshold:
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

    # Getting averages
    if numberOfXY1 > 0:
        avgX1 = totalX1 / numberOfXY1
        avgY1 = totalY1 / numberOfXY1
    
    if numberOfXY2 > 0:
        avgX2 = totalX2 / numberOfXY2
        avgY2 = totalY2 / numberOfXY2

    if avgX1 - avgX2 > width * avgDistThreshold and maxX - minX > width * minMaxDistThreshold: # if the average contours are separated enough
        image = cv2.circle(image, (int(avgX1), int(avgY1)), 20, (255,0,0), 3)
        image = cv2.circle(image, (int(avgX2), int(avgY2)), 20, (255,0,0), 3)
        peopleInFrame = 2

    elif avgX1 + avgX2 != 0 and avgY1 + avgY2 != 0 and numberOfXY1 + numberOfXY2 > 0: # if there are contours
        image = cv2.circle(image, (int((avgX1 + avgX2) / (2)), int((avgY1 + avgY2) / (2))), 20, (255,0,0), 3)
        peopleInFrame = 1

    else: # no contours
        peopleInFrame = 0


    cv2.imshow('frame', image)

    data = stream.read(CHUNK)
    rms = audioop.rms(data, 2)    # here's where you calculate the volume

    # parsing human data
    rememberdAudio.insert(0,rememberdAudio.pop(-1))
    rememberdAudio[0] = rms

    rememebrdPeople.insert(0,rememebrdPeople.pop(-1))
    rememebrdPeople[0] = peopleInFrame

    if talkDelay < lastTalkTime:
        ret = talk(round(np.average(rememebrdPeople)), np.average(rememberdAudio))
        if ret:
            lastTalkTime = 0

    time.sleep(powerUsageThrotle) 
    frameTimer = time.time() - startTime
    lastTalkTime += frameTimer

cap.release()
cv2.destroyAllWindows()

stream.stop_stream()
stream.close()
pyAud.terminate()