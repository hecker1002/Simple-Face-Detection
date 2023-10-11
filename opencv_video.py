import  cv2
import numpy as np

# Video pocessing using Opencv

# use pascal case in visdeo functiopn

# to play a video (collection of images in a sequence) , we use
# cv2.VideoCapture(index of camera(if live record) , input video file string ) method of Opencv .
# it also analyses the video using numpy
# and store the video in a ref var in python

# in opencv , cap = video capture

# to run a stored media file - try to keep in same dir as code
# cap = cv2.VideoCapture("sample.mp4")
cap = cv2.VideoCapture(0)

# cap.isOpened() =>in live web came video(index =0 , to see if camera is running or not
# return True / False

if cap.isOpened()==False :
    print("error displaying video from webcam")

# if camera /input media  file is working

# to read entire video file , read it frame-by-frame until its completed

# cap.read()   where cap = video -> reads entire vidso file frame-by-frame
# returns ret(bool) and frame(image - numpy 2D / 3D array)
# ret = bool shows wehter curr. frame was succesfully read or not
# frame = the 2D /3D np array =- image / curr frame to be displayed / analysed

# do this reading

# to save athe frames of recorded / operated  video on disk memory
# use video writer method of cv2 , and defien codec first =
# codec - compresion-decompresio techno. that tells multimedia soft. how to analu=yze the data into video files
# Use fourcc code for DIVX

fourcc = cv2.VideoWriter_fourcc(*'XVID') # saved file  =.avi format
# cv2.VIdeoWriter(output file name ,fourcc codec , fps(20) , frame size(width , height) format of output video file shold be
out = cv2.VideoWriter("output.avi" , fourcc , 21 , (800 , 800)) # its just a temporary memory loc top store processed video frame s
while cap.isOpened()==True :

    ret , frame = cap.read()

    if ret==True :
        # process the video files and save its frame colectively
        # straigthen the image by median blur
        frame_1 = cv2.medianBlur(frame , 5)
        frame_2 = cv2.GaussianBLur(frame , (5 , 5) , 0)

        frame1 = cv2.addWeighted(frame_1 , 0.5 , frame_2 , 0.5)

        img = cv2.cvtColor(frame1 , cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(img , 30 , 200)
        contours , heirarchy = cv2.findContours(edges , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE)
        # to write on frame s - writes on original frame , cv2.rectangle(frame , start , end , color , thick)
        cv2.rectangle(img , (100 , 100) , (210 , 210) , (0 , 255 , 255) , 3)
        # put text on video - cv2.putText(frame , string , start , font , fontsize , color , thickness 0 - changes original frame
        cv2.putText(img , "OPencv " , (100 , 100) , cv2.FONT_HERSHEY_SIMPLEX  , 1 , (0 , 255 , 255) , 3)
        cv2.drawContours(frame, contours , -1 , (0, 255, 0) , 1)
        # for face detection , use Haar Cascade Classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # detect faces in grayscale
        faces = face_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame , "Face Detected!!" , (x , y) ,cv2.FONT_HERSHEY_SIMPLEX  ,1 , (0 , 0 , 255) , 2)
        print(faces)
        cv2.imshow("current frame " , frame)
        # save each frame in output video file declared at start - by - out.write(frame)
        out.write(img)

 # while showing frame (image) , wait for 25 millisecond and if pressed key is - 'q' - ASCII(ord) code of presed key(0xFF) , then break  the while loop an dend the video
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else : # if ret = false , it means , file not read succesfuly , so break the loop
        break
     #if frame succesfuly read , display the frame

# to release system memory from ref var where video was stored
cap.release()

# release space from temporary out file as the actual procese fiel has been saved on sidk as output.avi
out.release()
cv2.destroyAllWindows()

'''' 
Frames - particular instances of video = images / 2D/3D matrix
multiple frames can pass in 1 sec in video

every operation that can be performed on img , can also be performed on frames
'''



