import cv2
from importlib import reload
import helper as h
# photometric interpretation: 'YBR_FULL_422'
#  Sort out paths....
relpath = "dicom_images"
# path_sobel =
# path_laplace =
reload(h)
ls_files = h.files(relpath)
img, img_grey = h.read_dcm(ls_files[1])


# Image Preprocessing
# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(img_grey, (9, 9), 0)
# Sobel Edge Detection
# h.sobel_edge_detection(img_blur, path_sobel)
# Alternatively Laplace Filter
img_laplace = h.laplacian_edge_detection(img_blur)


#  Edge Detection
# h.CannyEdgeDetection(img_blur)
# path_canny =
img_canny = h.CannyEdgeDetection(img_laplace).auto_canny()


# # create trackbar with contrast
# alpha = 1.3 # Contrast control (1.0-3.0)
# beta = 0 # Brightness control (0-100)
# adjusted = cv2.convertScaleAbs(img_gray, alpha=alpha, beta=beta)
# h.plot_before_and_after(img_gray, adjusted)

