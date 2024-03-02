import colorsys
import math
import numpy as np
import cv2


#image = cv2.imread('your_image.jpg')
cap = cv2.VideoCapture(0)

# Define the font for displaying text on the frame
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
   
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
# Convert the image to the HSV color space
    image_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green=np.array([74,204,148])
    upper_green=np.array([111,306,222])
    
    lower_red=np.array([138,145,192])
    upper_red=np.array([207,218,255])

    lower_blue=np.array([86,199,185])
    upper_blue=np.array([129,299,255])


# Create masks for red, blue, and green color ranges
    red_mask = cv2.inRange(image_hsv, lower_red, upper_red)
    blue_mask = cv2.inRange(image_hsv, lower_blue, upper_blue)
    green_mask = cv2.inRange(image_hsv, lower_green, upper_green)

# Calculate the percentage of red, blue, and green pixels in the image
    total_pixels = frame.shape[0] * frame.shape[1]
    red_percentage = np.sum(red_mask) / total_pixels
    blue_percentage = np.sum(blue_mask) / total_pixels
    green_percentage = np.sum(green_mask) / total_pixels

# Determine the dominant color based on the highest percentage
    
    dominant_color = None
    if red_percentage > blue_percentage and red_percentage > green_percentage:
     dominant_color = "Red"
    elif blue_percentage > red_percentage and blue_percentage > green_percentage:
     dominant_color = "Blue"
    elif green_percentage > red_percentage and green_percentage > blue_percentage:
     dominant_color = "Green"
    else :
      dominant_color = "Black"
    print("Dominant color:", dominant_color)
    if(dominant_color=="Red"):
     prev_pt = None
     pts = []
     contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over the contours
     for cnt in contours:
        # Get the area of the contour
        area = cv2.contourArea(cnt)

        # Only process contours with a minimum area
        if area > 500:
            # Get the centroid of the contour
            M = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # Draw a circle at the centroid of the contour
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            # If this is the first frame or the finger has moved significantly, set the previous point
            if prev_pt is None or abs(prev_pt[0] - cx) > 30 or abs(prev_pt[1] - cy) > 30:
                prev_pt = (cx, cy)

                # Add the current point to the list
                pts.append(prev_pt)

            # If the finger has not moved significantly, add the current point to the list
            if abs(prev_pt[0] - cx) <= 30 and abs(prev_pt[1] - cy) <= 30:
                prev_pt = (cx, cy)

                # Add the current point to the list
                pts.append(prev_pt)
        cv2.imshow("Video", frame)

    # Draw the full drawing on the frame
     if len(pts) > 1:
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), 2)

    # Show the video stream with the full drawing
    elif (dominant_color=="Green"):
       counter=0
       scale=50
       height, width, channels = frame.shape
       centerX,centerY=int(height/2),int(width/2)
       radiusX,radiusY= int(scale*height/100),int(scale*width/100)

       minX,maxX=centerX-radiusX,centerX+radiusX
       minY,maxY=centerY-radiusY,centerY+radiusY
       cropped = frame[minX:maxX, minY:maxY]
       resized_crop = cv2.resize(cropped, (width, height)) 
       counter=counter+1
       kernel = np.ones((5, 5), np.uint8)
       mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
       mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
       contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       pos = []
       for cnt in contours:
         area = cv2.contourArea(cnt)
         if area > 500: 
           
                moments = cv2.moments(cnt)
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                pos.append((cx, cy))
                # Draw a point at the centroid of the contour
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
         
    # If two green finger tapes have been detected, calculate the distance between them
       if len(pos) == 2:
        # Calculate the Euclidean distance between the two points
        dist = math.sqrt((pos[0][0] - pos[1][0]) ** 2 + (pos[0][1] - pos[1][1]) ** 2)
        # If the previous position of the green finger tapes is available, compare the current distance to the previous distance
        if(counter%100==0):
         
         if dist >250:
            scale-=5 #zoom in
            # Zoom out of the video
           
            print("Zoom in")
         elif dist <100:#60
            # Zoom in to the video
            scale+=5 #zoom out
           
            print("Zoom out")

       cv2.imshow("Video", frame)
    elif (dominant_color=="Blue"):
       print("capture heree")
       kernel = np.ones((5, 5), np.uint8)
       mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel, iterations=2)
       mask = cv2.dilate(blue_mask, kernel, iterations=1)

    # Find contours in the mask
       contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if two fingers are combined
       if len(contours) >= 2:
        fingers_combined = True
       else:
        fingers_combined = False

    # Draw contours on the frame
       cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
       if fingers_combined:
        cv2.imwrite("fingers_combined.jpg", frame)
        print("Image saved!")

    cv2.imshow("Video", frame)
       
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

