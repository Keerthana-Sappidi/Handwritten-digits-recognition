import cv2

# Load an image
img = cv2.imread(r'C:\Users\supriya\OneDrive\Desktop\digit4.png')

# Create a named window
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)

# Display the image
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()