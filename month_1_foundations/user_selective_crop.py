import cv2
 
# Load the image
img = cv2.imread("/home/shawncheng/Downloads/Image.png")
 
# Let user select ROI (drag a box)
roi = cv2.selectROI("Select ROI", img, False)
 
# Extract cropped region
cropped_img = img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
 
# Save and display cropped image
cv2.imwrite("/home/shawncheng/Downloads/Cropped_Image.png", cropped_img)  # Save the cropped image
cv2.imshow("Cropped Image", cropped_img)  # Display the cropped image
cv2.waitKey(0)  # Wait for a key press to close the displayed image
cv2.destroyAllWindows()