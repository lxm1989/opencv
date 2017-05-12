import cv2
face_patterns=cv2.CascadeClassifier('/home/guest/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
#sample_image=cv2.imread('/home/guest/Desktop/timg.jpg')
sample_image=cv2.imread('/home/guest/Desktop/1.jpg')
faces=face_patterns.detectMultiScale(sample_image,scaleFactor=1.1,minNeighbors=1,minSize=(1,1))
print("{0} faces was found.".format(len(faces)))
for(x,y,w,h) in faces:
    cv2.rectangle(sample_image,(x,y),(x+w,y+h),(0,255,0),2)
cv2.imwrite('/home/guest/Desktop/2.jpg',sample_image)