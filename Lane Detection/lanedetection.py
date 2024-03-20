import cv2
import matplotlib.pyplot as plt
import numpy as np

def grey(image):
    image=np.asarray(image)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def gaussianblur(image):
    return cv2.GaussianBlur(image, (5,5), 0)

def cannyedge(image):
    edges=cv2.Canny(image, 50, 150)
    return edges

def region(image):
    height, width = image.shape 

    triangle = np.array([
                       [(0, 90), (100, 50), (220,50), (width, 90)]
                       ])
    #print(triangle)
    mask = np.zeros_like(image)

    mask = cv2.fillPoly(mask, triangle, 255)
    mask = cv2.bitwise_and(image, mask)
    return mask

def average(image, lines):
    left = []
    right = []
    for line in lines:
        print("Line:")
        print(line)
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        y_int = parameters[1]
        if slope < 0:
            left.append((slope, y_int))
        else:
            right.append((slope, y_int))

    right_avg = np.average(right, axis=0)
    print("right:")
    print(right)
    left_avg = np.average(left, axis=0)
    left_line = make_points(image, left_avg)
    right_line = make_points(image, right_avg)
    return np.array([left_line, right_line])

def make_points(image, average):
    print("Average:")
    print(average)
    slope, y_int = average
    y1 = image.shape[0]
    #how long we want our lines to be --> 3/5 the size of the image
    y2 = int(y1 * (4/5))
    #determine algebraically
    x1 = int((y1 - y_int) // slope)
    x2 = int((y2 - y_int) // slope)
    return np.array([x1, y1, x2, y2])

def display_lines(image, lines):
    lines_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
        return lines_image
    

image_path = r"center_2024_01_19_03_15_45_909.jpg"
image1 = cv2.imread(image_path)
plt.imshow(image1)
copy = np.copy(image1)
edges = cv2.Canny(copy,50,150)
isolated = region(edges)
cv2.imshow('edges', edges)
cv2.imshow('isolated', isolated)
cv2.waitKey(0)
lines = cv2.HoughLinesP(isolated, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
averaged_lines = average(copy, lines)
black_lines = display_lines(copy, averaged_lines)
#taking wighted sum of original image and lane lines image
lanes = cv2.addWeighted(copy, 0.99, black_lines, 1, 1)
cv2.imshow('lanes', lanes)
cv2.waitKey(0)