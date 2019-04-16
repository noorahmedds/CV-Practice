import numpy as np
import cv2 as cv
import math

def getExtremaForScale(dog_scales, o, s):
	# Where o is the octave we are currently looking at and s is the scale inside that octave which is currently ours
	# Where i want to compare for dog_scales[o][s]
	# Between that and dog_scales[o][s-1] and dog_scales[o][s+1]
	src = cv.copyMakeBorder(dog_scales[o][s],1,1,1,1, cv.BORDER_CONSTANT,value=0)
	down = np.zeros(src.shape())
	up = np.zeros(src.shape())
	down_empty = True
	up_empty = True

	if (s > 0):
		down = cv.copyMakeBorder(dog_scales[o][s-1],1,1,1,1, cv.BORDER_CONSTANT,value=0)
		down_empty = False

	if (s < len(dog_scales[o])-1):
		up = cv.copyMakeBorder(dog_scales[o][s+1],1,1,1,1, cv.BORDER_CONSTANT,value=0)
		up_empty = False

	dog_maxima = np.zeros(src.shape())
	dog_minima = np.zeros(src.shape())

	# Lets first find maxima
	for i in range(1, src.shape(0)-1):
		for j in range(1, src.shape(1)-1):
			curr = src[i][j]
			src_neighbourhood = np.ravel(src[i-1:i+1, j-1:j+1])
			up_neighbourhood = np.ravel(up[i-1:i+1, j-1:j+1])
			down_neighbourhood = np.ravel(down[i-1:i+1, j-1:j+1])
			# What i should do is traverse over all neighbours one by one and determine if its max or not. That is the most efficient
			# But currently i am not doing that
			conc = np.concatenate((src_neighbourhood, up_neighbourhood, down_neighbourhood))

			if (curr == np.max(conc)):
				dog_maxima[i][j] == 1

			if (curr == np.min(conc)):
				dog_minima[i][j] == 1

			
	dog_scales[o][s] = src
	if (down_empty == False):
		dog_scales[o][s-1] = down
	if (up_empty == False):
		dog_scales[o][s+1] = up

	return dog_maxima + dog_minima

def extremaDetection(original, dog_scales):
	dog_len = len(dog_scales)
	extrema_scales = []
	for j in range(dog_len):
		dog_octaves = dog_scales[j]
		dog_octave_length = len(dog_octaves)
		extrema_octave = []
		for i in range(dog_octave_length):
			extrema = getExtremaForScale(dog_scales, j, i)
			extrema_octave.append(extrema)

		extrema_scales.append(extrema_octave)
	return extrema_scales

			
			


def sift():
	# Set all global variables like s (# if s = 2 then ks would be (2^(1/2)), 2(2^(1/2)) and so on and this time we will have 5 images in the stack)
	s = 2
	sigma = 1.0
	k = math.pow(2, 1/s)
	octave_capacity = s + 3
	octave_count = 1

	# Read image
	im = cv.imread("input.jpg")
	original = cv.cvtColor(im, cv.COLOR_RGB2GRAY)

	# Now we create a stack of gaussian images such that we have s+3 gaussians with successively multiples of k
	# Another Realization. It's very important that you choose the right size for your gaussian kernel
	# This is because the gaussian kernel's size will determing how many neighbours will affect the smooting of the convolution area
	# The optimal kernel size if unknown so i am currently choosing an arbitrary size for the kernel. About 5*5. For the purpose of extracting keypoints i will still be looking at neighbourhoods of 3x3 in adjacent gaussian images
	
	# Gaussian scales contains a list of gaussian stacks
	gaussian_scales = []
	
	# Now we start calculating the gaussians. and appending to the stack such that the top of the stack contains the the least scale
	# i.e. the last multiple of K will be at the bottom of the stack. (It really would not matter how it goes)

	for j in range(octave_count):
		gaussian_stack = []
		for i in range(octave_capacity):
			sig = (i+1) * k * sigma
			curr_gauss = cv.GaussianBlur(original, (5, 5), sig)
			gaussian_stack.append(curr_gauss)
			# ====> Note: Should technically generate DoG scales here. But too lazy ;)
		gaussian_scales.append(gaussian_stack)


	# Now we need to perform difference of gaussians
	dog_scales = []
	for j in range(octave_count):
		dog_octave = [] #one should know that this would always be octave_capaity - 1
		for i in range(1, octave_capacity):
			# when i'm at current. I want a DoG which i derive from gauss_scales[j][i-1] - gauss_scales[j][i]
			curr_dog = gaussian_scales[j][i-1] - gaussian_scales[j][i]
			dog_octave.append(curr_dog)
		dog_scales.append(dog_octaves)
	
	# Now we have to write the algorithm which determines whether or not a pixel is a minima or maxima.
	# #To get maximas # While traversing to check whether my current pixel is maximum. If it is greater than a neighbour then the neighbour turns to -1. If found a greater pixel then turn self to -1. Complete the traversal to turn all pixels which are less than your current pixel to -1
        # For minimas #While traversing to check whether my current pixel is maximum. If it is greater than a neighbour then the neighbour turns to -1. If found a lesser pixel then turn self to 999. Complete the traversal to turn all pixels which are greater than your current pixel to 999
        # This should give you two arrays, Minimas and maximas. You now just have to merge them.
        # After merging you need to show these as keypoints 

	# So we have four DoGs
	# Extrema Detection
	


	print(dog_scales)

	# # Example
	# sample1 = cv.GaussianBlur(gray, (11,11), 0.1)
	# sample2 = cv.GaussianBlur(gray, (11,11), 10)
	# dog = sample1 - sample2
	# cv.imshow("Original", gray) 
	# cv.imshow("Scale Space First Multiple: ", sample1) 
	# cv.imshow("Scale Space Second Multiple: ", sample2) 
	# cv.imshow("Difference of Gaussians: ", dog) 

	# cv.waitKey(0)
	# cv.destroyAllWindows()

	return

def main():
	"""SIFT Implementation: Based on https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf and http://aishack.in/tutorials/sift-scale-invariant-feature-transform-introduction/"""
	sift()
	return()
    # READ PAPER:
    # NOTES:
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
        # After merging you need to show these as keypoints



if __name__ == "__main__":
    main()