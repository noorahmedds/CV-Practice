import numpy as np
import cv2 as cv
import math
import matplotlib.pyplot as plt

# Legend:
# "====> Note: " Refers to refactoring notes

def generateGaussian(_sigma, _mu, shape=(10,10)):
	x, y = np.meshgrid(np.linspace(-1,1,shape[0]), np.linspace(-1,1,shape[1]))
	d = np.sqrt(x*x+y*y)
	sigma, mu = _sigma, _mu
	g = np.exp(-((d-mu)**2 / (2.0 * sigma**2)))
	return g

def getExtremaForScale(dog_scales, o, s):
	# Where o is the octave we are currently looking at and s is the scale inside that octave which is currently ours
	# Where i want to compare for dog_scales[o][s]
	# Between that and dog_scales[o][s-1] and dog_scales[o][s+1]
	src = cv.copyMakeBorder(dog_scales[o][s],1,1,1,1, cv.BORDER_CONSTANT,value=0)
	down = np.zeros(src.shape)
	up = np.zeros(src.shape)
	down_empty = True
	up_empty = True

	if (s > 0):
		down = cv.copyMakeBorder(dog_scales[o][s-1],1,1,1,1, cv.BORDER_CONSTANT,value=0)
		down_empty = False

	if (s < len(dog_scales[o])-1):
		up = cv.copyMakeBorder(dog_scales[o][s+1],1,1,1,1, cv.BORDER_CONSTANT,value=0)
		up_empty = False

	dog_maxima = np.zeros(src.shape)
	dog_minima = np.zeros(src.shape)

	# Lets first find maxima
	for i in range(1, src.shape[0]-2):
		for j in range(1, src.shape[1]-2):
			curr = src[i][j]
			src_neighbourhood = src[i-1:i+1, j-1:j+1]
			up_neighbourhood = up[i-1:i+1, j-1:j+1]
			down_neighbourhood = down[i-1:i+1, j-1:j+1]
			# ====> Note: What i should do is traverse over all neighbours one by one and determine if its max or not. That is the most efficient
			# But currently i am not doing that
			conc = np.vstack((src_neighbourhood, up_neighbourhood, down_neighbourhood))

			if (curr == np.max((src_neighbourhood, up_neighbourhood, down_neighbourhood))):
				dog_maxima[i, j] = 255

			if (curr == np.min((src_neighbourhood, up_neighbourhood, down_neighbourhood))):
				dog_minima[i][j] = 255
			


			
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
		for i in range(0, dog_octave_length):
			extrema = getExtremaForScale(dog_scales, j, i)
			extrema_octave.append(extrema)
			# cv.imshow("extrema_sample", extrema)
			# cv.waitKey(0)


		extrema_scales.append(extrema_octave)
	return extrema_scales

			
			
def generateLocalHistogram(src, i, j, g, n_size=(5,5)):
	# where src[i,j] is the keypoint. g is the gaussian kernel
	# We need to traverse a local neigbourhood of keypoint. n_szie(5,5)
	# generate a histogram with 36 bins and return them
	# first we generate the orientation and 
	hist_elements = []
	for ii in range(-2, 3):
		for jj in range(-2,3):
			_i = i + ii
			_j = j + jj
			m = math.sqrt(math.pow(src[_i+1, _j]-src[_i-1, _j],2) + math.pow(src[_i, _j+1]-src[_i, _j-1],2)) #this could be done with a kernel
			theta = math.tanh((src[_i, _j+1]-src[_i, _j-1])/(src[_i+1, _j]-src[_i-1, _j]))
			hist_elements.append(m * theta * g[ii+2, jj+2])
	# print(hist_elements)
	return np.histogram(hist_elements, bins=np.arange(36))

def sift():
	# Set all global variables like s (# if s = 2 then ks would be (2^(1/2)), 2(2^(1/2)) and so on and this time we will have 5 images in the stack)
	s = 2 #It should be noted that an octave should have 3 samplings
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
		dog_scales.append(dog_octave)
	print("Generated Difference of Gaussians")
	
	# Now we have to write the algorithm which determines whether or not a pixel is a minima or maxima.
	# #To get maximas # While traversing to check whether my current pixel is maximum. If it is greater than a neighbour then the neighbour turns to -1. If found a greater pixel then turn self to -1. Complete the traversal to turn all pixels which are less than your current pixel to -1
        # For minimas #While traversing to check whether my current pixel is maximum. If it is greater than a neighbour then the neighbour turns to -1. If found a lesser pixel then turn self to 999. Complete the traversal to turn all pixels which are greater than your current pixel to 999
        # This should give you two arrays, Minimas and maximas. You now just have to merge them.
        # After merging you need to show these as keypoints 

	# So we have four DoGs
	# Extrema Detection
	extrema_scales = extremaDetection(original, dog_scales)
	print("Detected Extremas")

	# ============= We will ignore implementation of 3.2 and 3.3

	# ============= 4: Now we perform removal of extrema responses. Out of all the extremas we calculated we need to reject extrema points which have a value of D[x] < 0.03 where image pixel is between 0, 1 or if we are looking at 0,255 then < 7. D[xhat] = D + 0.5(partialDeriv(D)' / partialDeriv(x))xhat
	
	# ============= 4.1: A poorly defined keypoint in the extrema will have a large prinicipal of curvature. (That is keypoint on a straight edge). read up on prinicipal Curvature. Create the Hessian Matrix (Yeh kis tarah banana hai). Determine Tr(H)^2/Det(H) < (r+1)^2/r where r = 10. If this is true then we keep the extrema otherwise we dont keep the extrema. As the ratio of the prinicipal curvate is greater than 10. Basically The ratio will be great if the prinicipal curvature is large across the edge and small to its perpendicular. This will mean that the keypoint was poor and probably on a straight edge.
	# To caluclate a Hessian matrix we need to make a derivative of the image and then another derivitive in each individual direction.
	extrema_sample = extrema_scales[0][3]
	gaussian_sample = gaussian_scales[0][3]
	gx = cv.GaussianBlur(extrema_sample,(3,1), 1)
	gy = cv.GaussianBlur(extrema_sample,(1,3), 1)

	gxx =  cv.GaussianBlur(gx, (3,1), 1)
	gxy =  cv.GaussianBlur(gx, (1,3), 1)

	gyx = cv.GaussianBlur(gy, (3,1), 1)
	gyy = cv.GaussianBlur(gy, (1,3), 1)

	# No we traverse all four arrays and get the intermediate 2x2 array which we use to find trace and Det
	# Tr(H) = Dxx + Dyy = α + β
	# Det(H) = DxxDyy − (Dxy)2 = αβ.

	# cv.imshow("Extrema Sample", extrema_sample)

	for i in range(gxx.shape[0]):
		for j in range(gxx.shape[1]):
			hessian = np.array([[gxx[i,j],gxy[i,j]],[gyx[i,j], gyy[i,j]]])
			trace = np.trace(hessian)
			det = np.linalg.det(hessian)
			if (math.pow(trace, 2)/det > 10):
				# eliminate this point
				extrema_sample[i, j] = 0

	print("Removed Extraneous Keypoints")
	cv.imshow("Fixed Sample", extrema_sample)

	# ============== Here goes the rest of the code ============= Orientation Assignment: regarding orientation assigning and final orientation and processing
	# For orientation we first need to store a window of sample points around a keypoint
	# for these sample points we create a histogram which will have 36 bins for 360 degrees.
	# The hist will contain orinetations such that each element of the histogram will represent the following
		# element = m(x,y) * theta(x,y) * gaussian_weight(x,y) 
		# (The gaussian weight basically is the gaussian weight from a circular gaussian kernel that is centered on the keypoint with a sigma 1.5 times the current scale)
		# The circular kernel should be of the radius used for the kernels above
	
	gaussian_weights = generateGaussian(1.0, 0, shape=(5,5)) #this will have variable sigma obviously
	extrema_sample = cv.copyMakeBorder(extrema_sample, 4,4,4,4, cv.BORDER_CONSTANT, value=0) #we pad the image with 4 pixels on each side for ease of coding
	gaussian_sample = cv.copyMakeBorder(gaussian_sample, 8,8,8,8, cv.BORDER_CONSTANT, value=0)

	print("Extrema Shape: ", extrema_sample.shape)
	print("Gaussian Shape: ", gaussian_sample.shape)

	hists = []
	for i in range(5, extrema_sample.shape[0]-5):
		for j in range(5, extrema_sample.shape[1]-5):
			if extrema_sample[i, j] == 255:
				# keypoint detected
				hists.append(generateLocalHistogram(gaussian_sample, i, j, gaussian_weights))

	plt.hist(hists[10], bins='auto')
	plt.show()


	# ============== Local Image Descriptor


	# # Example
	# sample1 = cv.GaussianBlur(gray, (11,11), 0.1)
	# sample2 = cv.GaussianBlur(gray, (11,11), 10)
	# dog = sample1 - sample2
	# cv.imshow("Original", gray) 
	# cv.imshow("Scale Space First Multiple: ", sample1) 
	# cv.imshow("Scale Space Second Multiple: ", sample2) 
	# cv.imshow("Difference of Gaussians: ", dog) 

	cv.waitKey(0)
	cv.destroyAllWindows()

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