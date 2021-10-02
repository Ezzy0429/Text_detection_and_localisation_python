#Group members
#Ahsan Shahid Minhas
#Chan Weng Keit
#Eh Zheng Yang
#Leong Wen Hao

#import all libraries
import numpy as np
import cv2

#prompt user for input
filename = input("Please input file name (e.g.: abc.jpg): ")
user_iteration = int(input("Input iteration (numbers only): "))
user_area = int(input("Input area(numbers only): "))
	
#read image as color
image = cv2.imread(filename, 1)

#convert image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#apply bilateral filter to grayscale image 
blur_image = cv2.bilateralFilter(gray_image,9,75,75)


#create kernel 
kernel = np.array([[-1, -1, -1], 
					[-1, 9, -1], 
					[-1, -1, -1]])
						
#apply image enhancement method to blur image
enhanced_image = cv2.filter2D(blur_image, -1, kernel)

#apply adaptive threshold to enhanced image
binary_image = cv2.adaptiveThreshold(enhanced_image, 255, 
									cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
									cv2.THRESH_BINARY_INV, 11, 30)
	
#create structuring element (kernel)
kernel_sE = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

#closing 
closed_image=cv2.morphologyEx(binary_image,cv2.MORPH_CLOSE,kernel_sE, iterations=user_iteration)

#find contours
#cv2.RETR_EXTERNAL to ignore contours inside the letters
#CV2.CHAIN_APPROX_SIMPLE to removes all redundant points 
#and compresses the contour to save memory
contours = cv2.findContours(closed_image,
							cv2.RETR_EXTERNAL, 
							cv2.CHAIN_APPROX_SIMPLE)

if len(contours) == 2:
	contours = contours[0]

else:
	contours = contours[1]

#initialize contour color and thickness 
color = (0,0,255)
thickness = 2

#print each contour
for each_contour in contours:
	area =  cv2.contourArea(each_contour)
	
	#set size of the area of white space to draw the contour 
	if area > user_area:
		#x,y is top-left coordinate of rectangle
		x, y, width, height = cv2.boundingRect(each_contour)
		cv2.rectangle(image, (x,y), 
					(x + width, y + height), 
					color, thickness)

#show output image
cv2.imshow("Original Image",gray_image)
cv2.imshow("Bilateral Filter",blur_image)
cv2.imshow("Enhanced Image", enhanced_image)
cv2.imshow("Adaptive threshold binary", binary_image)
cv2.imshow("Closed image", closed_image)
cv2.imshow("Final image", image)

cv2.waitKey()
cv2.destroyAllWindows()

