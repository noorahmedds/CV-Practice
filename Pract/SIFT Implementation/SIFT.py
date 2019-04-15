import numpy as np
import cv2 as cv

def main():
    """SIFT Implementation: Based on https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf and http://aishack.in/tutorials/sift-scale-invariant-feature-transform-introduction/"""
    
    # READ PAPER:
    # NOTES:
    im = cv.imread("input.jpg")
    gray = cv.cvtColor(im, cv.COLOR_RGB2GRAY)
    sift = cv.xfeatures2d.SIFT_create()
    kp = sift.detect(im) 
    cv.imshow(cv.drawKeypoints(gray, kp))
    cv.waitKey(0)

    # First we need to find keypoints stated in paper as scale space extremas
    # Selection of keypoints
    # Then we need to find orientations of each local point and rotate to find feature vectors for that locally 
    # Finally create the descriptor

    # Remember a gaussian can be decomposed into components
    # Basically what we need to do is 
    # For a particular ocatve (set of gaussian scales images):
        # We need to create atleast s+3 gaussian images where each gaussian image from the next has a sigma which is multiple of k
        # k is such that k = (2^(1/s)) so where s = 1; k would be 2, 4, 8, 16 and so on. and in we will have 4 blurred images in the stack
        # if s = 2 then ks would be (2^(1/2)), 2(2^(1/2)) and so on and this time we will have 5 images in the stack
        # We will have s+3 - 1 difference of gaussian images
        # Use the function cv.GaussianBlur(im, 3x3 = 9, sigmaX, sigmaY=0) sigmaX = k
        # For the next octave resample the gaussian image from the top of the stack of sigma

        # Local Extrema Detetction: For a particular DoG image. Choose a pixel iteratively and check its 8 neightbours (9x9 neighbourhood)
        # and the negihbours of its adjacent DoG images. If the current pixel is a maximum or minimum from its neighbours then that is considered on keypoint
        # Remember that to reduce processing time you need to remember that you are not rechecking the pixel which you have already checked whether its extrema or not in the previous scale
        # So what i can do is: if i my current pixel is not a local maxima, i turn its gray value to -1. If my current pixel is an extrema then i turn all the neighbouring pixels in the current scale and the scale above to -1
        #To get maximas # While traversing to check whether my current pixel is maximum. If it is greater than a neighbour then the neighbour turns to -1. If found a greater pixel then turn self to -1. Complete the traversal to turn all pixels which are less than your current pixel to -1
        # For minimas #While traversing to check whether my current pixel is maximum. If it is greater than a neighbour then the neighbour turns to -1. If found a lesser pixel then turn self to 999. Complete the traversal to turn all pixels which are greater than your current pixel to 999
        # This should give you two arrays, Minimas and maximas. You now just have to merge them.



if __name__ == "__main__":
    main()