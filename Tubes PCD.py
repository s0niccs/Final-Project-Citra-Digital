#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2


# In[2]:


faceCascade = cv2.CascadeClassifier("C:\\Users\\HP\\OneDrive\\Documents\\PCD/haarcascade_frontalface_default.xml")
noseCascade = cv2.CascadeClassifier("C:\\Users\\HP\\OneDrive\\Documents\\PCD/Nariz.xml")


# In[ ]:


video_capture = cv2.VideoCapture(0)
mask_on = False

while True:
    # ambil frame by frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    wajah = faceCascade.detectMultiScale(gray, 1.1, 5)

    # gambar kotak di wajah
    for(x, y, w, h) in wajah:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        if mask_on:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(frame,'Mask On',(x, y), cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 255, 0), 5)
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
            cv2.putText(frame,'Mask Off',(x, y), cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 0, 255), 5)

    hidung = noseCascade.detectMultiScale(roi_gray, 1.18, 35,)
    for(sx, sy, sw, sh) in hidung:
        cv2.rectangle(roi_color, (sh, sy), (sx+sw, sy+sh), (255, 0, 0), 2)
        cv2.putText(frame,'Hidung',(x + sx,y + sy), 1, 1, (0, 255, 0), 1)
    
    if len(hidung)>0:
        mask_on = False
    else:
        mask_on = True
    
 
    cv2.putText(frame,'Jumlah Wajah : ' + str(len(wajah)),(30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0), 2)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:




