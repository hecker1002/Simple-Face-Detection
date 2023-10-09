
# OpenCV - open source Computer vision to analyze the images / videos and implement ML in it
# images - a 2D / 3D matrix of values (range(0,255) and 3 color chnanels for colored iamges -Red R , green G  blue B

# all identifier like funct names in Opencv (cv2 alias) uses - Camel case , waitKey , destroyAllWindows , cvtColor , etc .

# Opencv always uses numpy in background (since imagesa are jsut a 2D / 3D matrix of numbers represnting color comp. of BGR .

# Grayscale images - 1 clor channel and 2D matrix represinting intensity of grey in each pixel
# RGB images (stored as BGR) - use 3 color channels and is a 3D matrix reresenting color comp of 3 colors
# in each pixel .


# read an image using opencv and store it in some ref. variable since pythoni s dynakic in nature
#im = image in cv2
import cv2
import numpy as np
# cv2.imread(string_path , flag ) -> to read an image (as matrix of numbers)
# try to keep the img in same dir as code

img = cv2.imread("ref.jpg",cv2.IMREAD_COLOR)  #flag = 1
# img = cv2.imread("ref.jpg" , cv2.IMREAD_GRAYSCALE) # flag = 0
# img = cv2.imread("ref.jpg" , cv2.IMREAD_UNCHANGED / flag = -1) ->
# uses alpha channel (transparency channel) -> shows how opaque / transparent a pixel is


# flag -> form in which img is read and stored in memory -> BGR(IMREAD_COLOR) , (gray)IMREA_GRAYSCALE , (alpha)IMREAD_UNCHANED

# cv2.imshow(win , img_address) - to display img in window named win

# cv2.imshow("window" , img)

print(img) # shows the value of colour component()channel) of each pixel of image (as 2D / 3D matrix)

# cv2.imwrite("file_name.jpg" , img_var )-> to save image on disk , (on current dir )
cv2.imwrite("saved.jpg" , img )

#cv2.split(img)to split the image pixels into its distinct color components -with each part of image showing where did colors lie
# bgr - color space (way of shhowing color channels of imagge)
b , g , r = cv2.split(img) # in python , var take value in same sequence of declaration

# cv2.imshow("b" , b) # the lighter regions show which pixels had "blue" in most intense amount .

# arithmetic operations on image = arithm. operation on pixels of image
test = cv2.imread("city.jpg" , 1)
test1 = cv2.resize(test , (2200 , 1468 ))
cv2.imwrite("city1.jpg" , test1)

city_1 = cv2.imread("city1.jpg")
print(city_1.shape , " "  , img.shape)
# addition of corresponding pixels of 2 imges (by specifying weights of each image) -> to superimpose 1 image over another
# both shouls have same channels and spatial dim .

img_add = cv2.addWeighted(city_1 , 0.5 , img , 0.4 ,  0) # amount of light = 0 (gamma

# cv2.substract(img1 , img2 ) => subtract one image pixels vlaue from another
img_sub = cv2.subtract(city_1 , img)

# Logical operations on image

# logical AND
# img1 = cv2.bitwise_and(img , city_1 , mask =None) # mask is a 8 bit vlaue

# img1 = cv2.bitwise_or(img , city_1 , mask =None) # bitwise OR

# img1 = cv2.bitwise_xor(img , city_1 , mask =None) # bitwise XOR

img1 = cv2.bitwise_not(city_1 , mask =None) # bitwise not

# cv2.resizeWindow("res" ,img_sub.shape[:2])

# resizing the image  = INTERPOLATION , SINCE WE ESTIAMTE THE APPROPRIATE PIXELS B/W ANY 2 GIVEN PIXELS .

#  cv2.resize(img_var , output_dim_tuple , interpolation_method) =to resize a img to particular dimensions
# some ineterpolation method are  - cv2.INTER_LINEAR(default , CV2.INTER_CUBIC , cv@.INTER_aREA
# resize() of cv2 does NOT change original img , so always store the resized one in a ref var in python

img_half = cv2.resize(img , (900 , 900) , interpolation = cv2.INTER_LINEAR )  # tuple showing size has (width.heigh) , NOT h,w


# cv2.erosion(img , kernel))to erode (remove) boundaries of objects in foreground (things)
# kernel = filter matrix containing info of how much erosion and on what should be done .
kernel = np.ones((5,5) , np.uint8)
erode = cv2.erode(img , kernel )

# cv2.Blurring_techniques() => to reduce noises in an image and preserve imp features

# gausian blur - preserves edges and uses gaussian matrix as convolving opert (filter)
# cv2.GaussianBlur(img , size of gaussian filter matrix  , std. deviation(udually 0)
gauss_blur = cv2.GaussianBlur(img ,(5,5) , 0 )

# median blur - replaces pixel intensity = median(surr pixel intensities) = BEST FOR BLURRING
# cv2.medianBlur(img , one_dim_of_filter_size)
median_blur  = cv2.medianBlur(img , 5)

#cv2.BilateralFilter(img , diam of pixel nbd , std_dev in color space , std_dev in coordinate space )
bil_fil = cv2.bilateralFilter(img , 9 , 75 , 75)

# cv2.copyMakeBorder(img ,top , bottom left_b(in pixels) , right_b , botder type , vlaue  = color of border  )
border = cv2.copyMakeBorder(img , 10 , 10 , 10 , 10 , borderType = cv2.BORDER_CONSTANT ,value = 0)

# cv2.getRottioMatrix2D(center ,angle_degrees , scaling factor )to rotate an image => it gives matrix containing final position of rotaed basis vectors
# using above matrix , we do lin transformation on out matrix - cv2.warpAffine(img , rotationmatrix , ssize of final img )
rot_mat = cv2.getRotationMatrix2D((img.shape[0]//2 , img.shape[1]//2) , 30 , 1)
rot_img = cv2.warpAffine(img , rot_mat , (img.shape[0] , img.shape[1]))
# gray_img  = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY) OR use flag =0 in cv2.imread(path , flag)

# do normalization of pixels - bring all pixel intensity closer to each other .
_norm = cv2.normalize(img.astype('float'), None, 0, 1, cv2.NORM_MINMAX )

# cv2.line(img  , start , end , color , thickness) to draw a line on the image
img_line = cv2.line(img , (0,0) , (150 , 170) , (0, 0, 0) , 12)

# cv2.arrowedLine(img , s , e , c , t , tipLength )to draw an arrow over image
img_arrow = cv2.arrowedLine(img , (0,200) , (150 , 190) , (0, 0, 0) , 12 , tipLength = 0.5)

''' length lie thickness radius , etc  is measured in cv2 - opencv in pixels (px) '''
# cv2.circle(img , (center_x , centre_y) , radius , color , thickness)->to draw a circle over image -
img_circle = cv2.circle(img , (210 , 210) , 12 , (0 , 255 , 0) , 4)

# cv2.rectangle(img , start , end , color , thicknes )to draw a rectangle over wholwe image
img_rect = cv2.rectangle(img , (110 , 120) , (600 , 670) , (255 , 255, 0) , -1) # thiickness =-1 , fill thwe whole obj by same color

# to put somwhere in image(7 args) - give in third arg , coordinate of bootm-left corner of text to set position of text over img
img_text = cv2.putText(img , "OpenCV" , (500 , 700) , cv2.FONT_HERSHEY_SIMPLEX , 3 , color = (255 , 0 , 255 ) , thickness = 10 )

# In opencv , Canny edge detection  - is a DL model that finds edges of obj. in given image ->
# works ONLY on grayscale amges (512 , 512 , 1) - 1 color channel of gray intensity

# we use Canny edge detection - to find contours in grayscale img .

# load an img in grayscale mode
gray = cv2.imread("ref.jpg" , 0)

# find edges using canny edge detecttion cv2.Canny(img , lowet_threshold , upper_threshold)
# lower threshold to upepr threshol;d - stong pixeld which are included in voundary / edges in images
edge = cv2.Canny(gray , 30 , 200)

# edges in images = Contours - usually pixels of same intensity
# cv2.findContours(grayssale img , retrievel , approximation->(wether we want to store all points og boundary in contour / specifiv pts only))
# rturns a list of contours , heirarchy (relation b/w contours)

contours , heirarchy = cv2.findContours(edge , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE)
print("no of contours" , len(contours))
gray1 = cv2.cvtColor(gray , cv2.COLOR_GRAY2BGR)
# cv2.drawContours(img , contour list , -1(draw all contours) , color , thickness of contour lines )draw all extracted contours Onto origina img .
cv2.drawContours(gray1, contours , -1 , (0, 255, 0) , 3) # changes the original grayscale img , so just call the function , dont stoe in soem ref var

# to detect circles n image , use c2.Houghzircles(img , method of gradient , distance b/w circle , min max radius )
detected_circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, param2 = 30, minRadius = 1, maxRadius = 40)

# Draw detected circles
if detected_circles is not None: # they exist

    # Convert the circle parameters a, b and r to integers.
    detected_circles = np.uint16(np.around(detected_circles))

    for pt in detected_circles[0, :]:
        a, b, r = pt[0], pt[1], pt[2]
     # cnetr and radius of circle
        # Draw the circumference of the circle. using cv2,circle method
        cv2.circle(gray, (a, b), r, (0, 255, 0), 2)

''' its possible , contour lines != canny edges detected , since due to heirarchy ni contour , some could be merge or NOT show dur to low corr.'''
cv2.imshow("win" ,gray)



cv2.waitKey(0) # wait tilll user preses a key to close window
print("\n" , img.shape) # img.shape - width * height* no. of channels in cv2 (opencv
cv2.destroyAllWindows() # after closing , erase all space of window from memory



''' 
IMPORTANT - 

In open cv , img stored in BGR format and dimension has (width , height) , NOT (h , w) 
 Negatives of a reel / img = actually cv2.bitwise_not (bitwise NOT operation on image 
 
 Masking in Image processing - extracting ONLY certain parts of image 
 in cv2 (Opencv) , to use a color , pass a tuple of BGR format - Blue - (255, 0 ,0 ) , Black(0 , 0 ,0) , etc
 
 Coordinates of output window in cv2 starts from top-left = (0,0)
 
 In opencv , to produce a black image , we need a 3D array ( width , heigh , no of color channels) with each value in matrix - 
 each pixel shows ONLY color com[onent , spatial dim , we already gave , we set to 0 , so make a 3D array all filled with zeroes.
 np.zeroes((512 ,512 , 3) , dtype(np.uint8)
 # these cv2 methods work ONLY on copies ogf iamges
 
 We have to use cv2.findContours() even after usign Canny edge detection because - Canny edge detcion onlt DETECTS the edges
 But info. about features like shape , largness , smallnes sof all obh in kimages based on edges and relation b/w them (hierrch) 
 is given only by - cv2.findContours()
 
 if you are showing / drawing anythin obn image which are grayscale , any color fill finally become - black/white .
 '''
