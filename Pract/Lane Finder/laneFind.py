import numpy as np
import cv2 as cv

def readInput(fileHandle):
    video = cv.VideoCapture(fileHandle)
    # print (video)
    return video

def getLine(src, dest):
    # returns the gradient and intercept of the line connecting the two
    m = ((dest['y'] - src['y']) / (dest['x'] - src['x']))
    c = dest['y'] - (m * dest['x'])
    return m,c

def isOnLine(y, x, m, c):
    return y == (m*x + c)

def createMask(shape, point1, point2, point3):
    mask = np.zeros(shape, dtype = np.uint8)
    # point1 and point 2 are points on the bottom of the frame
    # point 3 is the vanishing point
    rows, columns = shape
    
    m, c = getLine(point1, point3)
    m2, c2 = getLine(point2, point3)
    print (m2, c2, m, c)
    for i in range(0, rows):
        for j in range(0, columns):
            if i > ((m*j) + c) and i > ((m2*j) + c2):
                mask[i, j] = 255
    return mask

def processFrame(frame, mask):
    # output contains the final lines after the hough transform

    kernel = np.ones((5,5))
    kernel = kernel/25

    # apply gaussian blur
    gauss = cv.filter2D(frame, 0, kernel)
    canny = cv.Canny(gauss, 50, 100)

    # now we create a mask to get only the front perspective of the image
    output = cv.bitwise_and(canny, canny, mask=mask)

    return output

def sortHoughLines(lines):
    rightLeaningRho = 0
    rightLeaningTheta = 0
    leftLeaningRho = 0
    leftLeaningTheta = 0

    rlCount = 0
    llCount = 0

    for line in lines:
        rho = line[0][0]
        theta = line[0][1]

        if (theta < 1.5):
            rightLeaningRho += rho
            rightLeaningTheta += theta
            rlCount+=1
        else:
            leftLeaningRho += rho
            leftLeaningTheta += theta
            llCount+=1

    rightLeaningRho = rightLeaningRho/rlCount
    rightLeaningTheta = rightLeaningTheta/rlCount
    leftLeaningRho = leftLeaningRho/llCount
    leftLeaningTheta = leftLeaningTheta/llCount

    return rightLeaningRho, rightLeaningTheta, leftLeaningRho, leftLeaningTheta


def showLine(p_frame, rho, theta):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv.line(p_frame, (x1, y1), (x2, y2), (100,0,0), 5)

def showVideo(video, mask):
    for i in range(0, 2):
        ret, frame = video.read()
        
        gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        # do processing on the given frame
        p_frame = processFrame(gray, mask)

        lines = cv.HoughLines(p_frame, 1, np.pi/180, 100)
        print("frame:  "+ str(i))
        print (lines)

        # Should sort the given lines based on opposing gradients and then find the most hit block to retain
        rrho, rtheta, lrho, ltheta = sortHoughLines(lines)

        showLine(p_frame, rrho, rtheta)
        showLine(p_frame, lrho, ltheta)

        cv.waitKey(0)

        cv.imshow('frame', p_frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv.waitKey(0)
    video.release()


# First lets read video frames from the input video

# Apply a simple gaussian filter to smooth out the image
# Apply both with a blur filter and by using imfilter

# Apply the canny edge detector with its particular thresholds

# Create a mask which ands the canny input so that only the view right infront of the car remains

# Return the hough transform hits and apply a threshold to determine two lines (obviously in opposing directions)
# As the parallel lines on the road are vanishing towards a single point

# Optional Estimate the shape that remains. I.e. create a mask which has the shape from the associated lines

def main():
    fileHandle = 'input.mp4'
    video = readInput(fileHandle)
    
    _, frame = video.read()
    frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    # points1, points2, points3
    # point = {'x':, 'y':} 
    point1 = {'x':78, 'y':479}
    point2 = {'x':750, 'y':479}
    point3 = {'x':384, 'y':299}

    # Remember when creating a particular mask, dont just use the shape instead make a copy of the same frame. Because of type errors. Very important for this purpose. The mask was initially not that.
    # The mask that numpy creates with ones of zeros is float64. So we need to be careful because out image is an uint8
    p_mask = createMask(frame.shape, point1, point2, point3)

    showVideo(video, p_mask)

    return

def houghExperiment(frame):
    lines = cv.HoughLines(frame, 1, np.pi/180, 200)
    cv.imshow("Hough Lines", lines)
    cv.waitKey(0)


if __name__ == "__main__":
    main();
    cv.destroyAllWindows();