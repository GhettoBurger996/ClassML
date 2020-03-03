# import the necessary packages
from skimage import feature
import skimage

class HOG:
	def __init__(self, orientations = 9, pixelsPerCell = (8, 8),
		cellsPerBlock = (3, 3), normalize = False, block_norm = "L1"):
		# store the number of orientations, pixels per cell,
		# cells per block, and whether or not power law
		# compression should be applied
		self.orientations = orientations
		self.pixelsPerCell = pixelsPerCell
		self.cellsPerBlock = cellsPerBlock
		self.normalize = normalize
		self.block_norm = block_norm

	def describe(self, image):
		# compute HOG for the image
		# compute Histogram of Oriented Gradients features for scikit-image < 0.13
		if int(skimage.__version__.split(".")[1]) < 13:
			hist = feature.hog(image, orientations=self.orientations, 
				pixels_per_cell=self.pixelsPerCell,
				cells_per_block=self.cellsPerBlock,
				transform_sqrt=self.normalize)

		# otherwise comput Histogram of Oriented Gradients features for scikit-image >= 0.13
		else:
			hist = feature.hog(image, orientations=self.orientations, 
				pixels_per_cell=self.pixelsPerCell,
				cells_per_block=self.cellsPerBlock,
				transform_sqrt=self.normalize,
				block_norm=self.block_norm)

		# return the HOG features
		return hist