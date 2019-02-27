
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import time


# In[14]:


cap = cv2.VideoCapture(0)
time.sleep(3)
count = 0
background = 0

for i in range(60):
    ret, background = cap.read()
background=np.flip(background,axis=1)

while(cap.isOpened()):
    ret, image = cap.read()
    if not ret:
        break
    count+=1
    image = np.flip(image, axis=1)
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    lower_range = np.array([110,50,50])
    upper_range = np.array([120,255,255])
    mask1 = cv2.inRange(hsv, lower_range, upper_range)
    
    lower_range = np.array([120,50,50])
    upper_range = np.array([130,255,255])
    mask2 = cv2.inRange(hsv, lower_range, upper_range)
    
    mask1 = mask1 + mask2
    
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations = 2)
    mask1 = cv2.dilate(mask1, np.ones((3,3), np.uint8), iterations=1)
    mask2 = cv2.bitwise_not(mask1)
    
    res1 = cv2.bitwise_and(background, background, mask = mask1)
    res2 = cv2.bitwise_and(image, image, mask = mask2)
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)
    
    cv2.imshow("hello",final_output)
    k= cv2.waitKey(10)
    if k==27:
        break
cv2.destroyAllWindows()
cap.release()

