import os
import os.path as osp
import re
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
from scipy import ndimage as ndi
from skimage.io import imread
from skimage import feature
from skimage import color
from collections import OrderedDict
import time



class ImgList():
	def __init__(self):
		self.workingimg = [] #current working image
		self.imgname = '' #name assigned by user
		self.filename = '' #name of file
		self.colorimg = [] #colored image
		self.startingimg = [] #placeholder for 
		self.filelist = [] #list of all files in directory
		self.imgtype = '' #RGB
		self.path = '' #path to main folder
		self.filenum = 0
		

	def setImagePath(self, path):
		self.path = path

	def setImageType(self, choice):
		self.imgtype = choice

	def getFileList(self):
		choice = self.imgtype
		mypath = self.path
		self.imgname = ''
		pattern = re.compile("\D\D\D_\D\D\D\D_\d*_" + choice + ".tif")

		files = [f for f in os.listdir(mypath) if osp.isfile(osp.join(mypath, f)) and pattern.match(f)]
		files = list(map(lambda x: mypath + x, files))
		files.sort()
		self.filelist = files
		return files

	def retrieveAll(self):
		files = self.getFileList()
		filelist = list(map(lambda x: misc.imread(x), files))
		self.startingimg = filelist[0]
		self.workingimg = filelist[0]
		return filelist

	def retrieveSingle(self, count=1, offset=0):
		files = self.getFileList()
		end = offset + count
		file = files[offset:end]
		filelist = list(map(lambda x: misc.imread(x), file))
		self.colorimg = filelist[0]
		self.filename = files[offset]
		return filelist

	def retrieveGrey(self, count=1, offset=0):
		files = self.getFileList()
		end = offset + count
		file = files[offset:end]
		filelist = list(map(lambda x: misc.imread(x, flatten=True), file))
		self.startingimg = filelist[0]
		self.workingimg = filelist[0]
		return filelist

	def applyDOG(self, sigma):
		sigma = int(sigma)

		rgb_dog = self.colorimg - ndi.gaussian_filter(self.colorimg, sigma=sigma)
		self.workingimg = color.rgb2gray(rgb_dog)
		print("...Difference of Gradients applied")

	def filterImage(self, filtername, sigma):
		grey = self.workingimg
		sigma = int(sigma)

		if(filtername == '1' or filtername == 'gauf'):
			greyg = ndi.gaussian_filter(grey, sigma=sigma)
			self.workingimg = greyg

		if(filtername == '2' or filtername =='gagm'):
			greyg = ndi.gaussian_gradient_magnitude(grey, sigma=sigma)
			self.workingimg = greyg
		#greyy = ndi.gaussian_filter(grey, sigma=(4,4,0))\
		
		# #
		# dog2 = color.rgb2gray(rgb_dog2)
		# greyf = ndi.gaussian_filter(dog2, sigma=3)
		
		####
		# Second filters
		#
		#greyy = ndi.gaussian_gradient_magnitude(dog, sigma=3)
		
		
		#edges3 = feature.canny(grey[0], sigma=8)

	def applyCanny(self, sigma):
		sigma = int(sigma)

		edges1 = feature.canny(self.workingimg, sigma=sigma)
		#edges2 = feature.canny(img2[0], sigma=3)

		print("...Canny edge detection complete")
		self.workingimg = edges1
		#return edges1

	def drawImage(self):
		print("...Drawing image")
		#display results
		fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 2),
		                                    sharex=True, sharey=True)

		ax1.imshow(self.colorimg, cmap=plt.cm.gray)
		ax1.axis('off')
		ax1.set_title('Color', fontsize=20)

		ax2.imshow(self.workingimg, cmap=plt.cm.gray)
		ax2.axis('off')
		ax2.set_title('Working', fontsize=20)
		plt.show()

	def convolve(self, kernel):
		from skimage.exposure import rescale_intensity
		# grab the spatial dimensions of the image, along with the spatial dimensions of the kernel
		image = self.workingimg
		laplacian = np.array((
			[0, 1, 0],
			[1, -4, 1],
			[0, 1, 0]), dtype="int")
		sobelX = np.array((
			[-1, 0, 1],
			[-2, 0, 2],
			[-1, 0, 1]), dtype="int")
		if(kernel == 'lapl'):
			kernel = laplacian
		elif(kernel == 'sobx'):
			kernel = sobelX
		else:
			return -1

		(iH, iW) = image.shape[:2]
		(kH, kW) = kernel.shape[:2]
	 
		# allocate memory for the output image, taking care to
		# "pad" the borders of the self.pt_input image so the spatial
		# size (i.e., width and height) are not reduced
		pad = int((kW - 1) / 2)
		image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
		output = np.zeros((iH, iW), dtype="float32")

		for y in np.arange(pad, iH + pad):
			for x in np.arange(pad, iW + pad):
				# extract the ROI of the image by extracting the *center* region of the current (x, y)-coordinates dimensions
				roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
 	
				# perform the actual convolution by taking the element-wise multiplicate between the ROI and the kernel, then summing the matrix
				k = (roi * kernel).sum()
 	
				# store the convolved value in the output (x,y)-coordinate of the output image
				output[y - pad, x - pad] = k
		# rescale the output image to be in the range [0, 255]
		output = rescale_intensity(output, in_range=(0, 255))
		output = (output * 255).astype("uint8")
 	
		# return the output image
		self.workingimg = output
		print(">>> Convolution applied...")
		#return output

	
class Program(ImgList):
	def __init__(self):
		ImgList.__init__(self)
		
		self.workinglist = {}
		self.loglist = [] #list of transformations applied to image
		self.manipulationlog = []
		self.lastcmd = ''

		self.function_map = {
			'load': self.load_Controller,
			'conv': self.conv_Controller,
			'filt': self.filterImage_Controller,
			'filter': self.filterImage_Controller,
			'drawimg': self.drawImage_Controller,
			'draw': self.drawImage_Controller,
			'canny': self.canny_Controller,
			'dog': self.dog_Controller,
			'current': self.current_Controller,
			'curr': self.current_Controller,
			'store': self.store_Controller,
			'save': self.store_Controller,
			'storeas': self.storeas_Controller,
			'saveas': self.storeas_Controller,
			'showlist': self.showImageList,
			'retrieve': self.retrieve_Controller,
			'use': self.retrieve_Controller,
			'reset': self.reset_Controller,
			'log': self.log_Controller,
			'viewlog': self.log_Controller,
			'quit': self.quit_Controller,
			'q': self.quit_Controller,
			'genlog': self.genlog_Controller,
			'help': self.help_Controller,
		}
		self.navfunc = ['q', 'quit', 'reset', 'use', 'retrieve', 'showlist', 'saveas', 'save', 'store', 'storeas', 'draw', 'drawimg', 'current', 'curr', 'log', 'viewlog', 'genlog', 'help', 'load']
		self.go = True


	################################################################################
	################################################################################
	####																		####
	####						Program functions 		 					 	####
	####																		####
	################################################################################
	################################################################################
	def storeImage(self, name):
		# if name in workinglist and name == False:
		# 	obj = {"work": self.workingimg, 'color': self.colorimg, 'filename': self.filename}
		# 	self.imgname = name
		# 	self.workinglist[name] = obj
		# 	print("...Image stored as: " + name)
		if name in self.workinglist:
			confirm = self.pt_input(">>> Are you sure you want to overwrite " + name + "? (y/n)\n", 'res')
			if confirm == 'y':
				obj = {"work": self.workingimg, 'color': self.colorimg, 'filenum': self.filenum, 'filename': self.filename, 'log': self.manipulationlog}
				self.imgname = name
				self.workinglist[name] = obj
				print("... Image stored as: " + name)
			else:
				print("... File NOT saved")
		else:
			obj = {"work": self.workingimg, 'color': self.colorimg, 'filenum': self.filenum, 'filename': self.filename, 'log': self.manipulationlog}
			#name = self.imgname
			self.imgname = name
			self.workinglist[name] = obj
			print("...Saved as " + name)

	def showImageList(self):
		i = 1
		print("...Saved images: ")
		self.workinglist = OrderedDict(sorted(self.workinglist.items(), key=lambda t: t[0]))
		for array in self.workinglist:
			print("  ", i, ".", array)
			i += 1

	def retrieveImage(self, choice):
		self.workinglist = OrderedDict(sorted(self.workinglist.items(), key=lambda t: t[0]))
		names = list(self.workinglist)
		#self.manipulationlog = []

		if choice in self.workinglist:
			self.imgname = choice
			self.workingimg = self.workinglist[choice]['work']
			self.colorimg = self.workinglist[choice]['color']
			self.filename = self.workinglist[choice]['filename']
			self.filenum = self.workinglist[choice]['filenum']
			self.manipulationlog = self.workinglist[choice]['log']
			print("...Image retrieved: " + self.filename)
			print("...Name: " + self.imgname)
		else:
			choice = int(choice)
			choice = choice - 1
			self.imgname = names[choice]
	
			if(self.workinglist[names[choice]]):
				self.workingimg = self.workinglist[names[choice]]['work']
				self.colorimg = self.workinglist[names[choice]]['color']
				self.filename = self.workinglist[names[choice]]['filename']
				self.filenum = self.workinglist[names[choice]]['filenum']
				self.manipulationlog = self.workinglist[names[choice]]['log']
				print("...Image retrieved: " + self.filename)
				print("...Name: " + self.imgname)
			else:
				print("...Error")

	def imageLog(self, func, tag):
		split = func.split(' ')
		if split[0] in self.navfunc:
			tag = 'nav'
		if tag == 'ias':
			cmdlength = self.lastcmd.split(' ')
			if len(cmdlength) > 1:
				self.lastcmd = func
				self.manipulationlog.append(self.lastcmd)
		elif tag == 'res':
			self.lastcmd += " " + func
		# elif tag == 'end':
		# 	self.lastcmd += " " + func
		# 	self.manipulationlog.append(self.lastcmd)

	def generalLog(self, func, tag):
		split = func.split(' ')
		if split[0] in self.navfunc:
			tag = 'nav'
		self.loglist.append(func + "|" + tag + "|" + str(time.time()))

	def pt_input(self, msg, tag):
		response = input(msg)
		split = response.split(' ')


		if tag == 'res':
			return response
			

		if split[0] in self.function_map and len(split) == 1:
			cmd = self.function_map[split[0]]
			cmd()

		elif split[0] in self.function_map and len(split) > 1:
			i = 0
			args = []
			for entry in split:
				if i == 0:
					i += 1
					continue
				args.append(entry)
				i += 1
			cmd = self.function_map[split[0]]
			cmd(*args)
		else:
			print("Command not recognized.  Type 'help' for detailed help")
			return -1

		self.imageLog(response, tag)
		self.generalLog(response, tag)

		#return [response, tag]

	def im_input(self, msg):
		response = input(msg)
		
		if response == 'log':
			pass
		else:
			return response
	################################################################################
	################################################################################
	####																		####
	####							Controllers			 					 	####
	####																		####
	################################################################################
	################################################################################
	def load_Controller(self, args=False):
		if args == False:
			fn = self.pt_input(">>> Enter a file number\n", 'res')
			fn = int(fn)
		else:
			fn = int(args)
		self.filenum = fn
		self.retrieveGrey(1, fn)
		self.retrieveSingle(1, fn)
		print("...Image loaded")
		print(self.filename)
		self.manipulationlog = []

	def conv_Controller(self, CONV_OPT=False):
		if CONV_OPT == False:
			CONV_OPT = self.pt_input(">>> Enter convolution:\n  1. Laplacian\n  2. Sobel-X\n", 'res')

		if(CONV_OPT == "1"):
			CONV_OPT = 'lapl'
		if(CONV_OPT == '2'):
			CONV_OPT = 'sobx'	
		callback = self.convolve(CONV_OPT)

		if callback == -1:
			print("...Error with convolve parameters")

	def filterImage_Controller(self, FILT_OPT=False, FILT_SIGMA=False):
		if FILT_OPT == False:
			FILT_OPT = self.pt_input(">>> Choose filter:\n  1. Simple Gaussian Blur\n  2. Gaussian Gradient Magnitude\n", 'res')
			FILT_SIGMA = self.pt_input(">>> Enter sigma value:\n", 'end')

		self.filterImage(FILT_OPT, FILT_SIGMA)
		print("...Applied filters")

	def drawImage_Controller(self):
		self.drawImage()

	def canny_Controller(self, CANNY_SIGMA=False):
		if CANNY_SIGMA == False:
			CANNY_SIGMA = self.pt_input(">>> Enter sigma value:\n", 'res')
		self.applyCanny(CANNY_SIGMA)

	def dog_Controller(self, DOG_SIGMA=False):
		if DOG_SIGMA == False:
			DOG_SIGMA = self.pt_input(">>> Enter sigma value:\n", 'res')

		self.applyDOG(DOG_SIGMA)

	def current_Controller(self):
		print("==========INFO==========")
		print("File number: ", self.filenum)
		print("Current image: ", self.filename)
		if(self.imgname):
			print("Name of image: ", self.imgname)
		else:
			print("Name of image: [no name]")
		print("Manipulation Log: ")
		for item in self.manipulationlog:
			print("  - " + item)
		print("=========================")

	def store_Controller(self, name=False):
		if name == False and self.imgname == '':
			name = self.pt_input(">>> Enter a name for this image:\n", 'res')
		elif name == False and self.imgname != '':
			name = self.imgname
		else:
			name = name
		self.storeImage(name)
	
	def storeas_Controller(self, name=False):
		if name == False:
			name = self.pt_input(">>> Enter a name for this image:\n", 'res')
		self.storeImage(name)
		
	def retrieve_Controller(self, RETR_OPT=False):
		if RETR_OPT == False:
			self.showImageList()
			RETR_OPT = self.pt_input(">>> Pick a saved image:\n", 'res')
		self.retrieveImage(RETR_OPT)

	def reset_Controller(self):
		confirm = self.pt_input("Are you sure you want to clear the working list? (y/n)\n", 'res')
		if(confirm == 'y'):
			print("List cleared")
			self.workinglist = {}

	def log_Controller(self):
		print("=========LOG MENU==========")
		for item in self.manipulationlog:
			print(item)

		print("===========================")

	def quit_Controller(self):
		confirm = self.pt_input(">>> Are you sure you want to quit?  Any unsaved progress will be lost (y/n)\n>", 'res')
		if(confirm == 'y'):
			print("===Goodbye===")
			self.go = False

	def genlog_Controller(self):
		print("=========LOG MENU==========")
		for item in self.loglist:
			print(item)

		print("===========================")

	################################################################################
	################################################################################
	####																		####
	####						Help / Message functions 					 	####
	####																		####
	################################################################################
	################################################################################
	def mainMessage(self):
		clear = lambda: os.system('clear')
		clear()
		print("=========MAIN MENU=========")
		print("  - Help Menu (help)")
		print("  - Load an image (load)")
		print("  - Manipulate photo (filt, dog, conv)")
		print("  - Draw image (drawimg)")
		print("  - Save image in workspace for future use (store, showlist, retrieve)")
		print("  - Quit (quit)")
		print("===========================")

	def help_Controller(self):
		go = True
		while go == True:
			clear = lambda: os.system('clear')
			clear()
			print("=========HELP MENU=========")
			print(" Welcome to the help menu.  Access one of the following...")
			print("   1. Getting Started")
			print("   2. Commands")
			print("   3. Tips")
			print("   4. Exit")
			print("===========================")

			choice = self.pt_input(">", 'res')

			if choice == '1':
				self.helpIntro()
			elif choice == '2':
				self.helpCommands()
			elif choice == '3':
				pass
			elif choice == '4':
				self.mainMessage()
				break
				return 0

	def helpIntro(self):
		clear = lambda: os.system('clear')
		clear()
		print("=========HELP MENU========")
		print("\nThis program is used for prototyping image preprocessing algorithms. Many possible combinations of filters and image transformations can be applied to images before applying an algorithm such as the Canny edge detector. Such pre-processing algorithms are used to increase uniformity, reduce dimensionality, reduce noise, and more which increases the overall effectiveness of edge detectors.")
		print("\nEdge detection is commonly used in image processing algorithms in order to identify and separate different elements of an image such as buildings and landscape or animal and background. In order for a computer to 'identify' the target, it needs to be able to detect the bounds of that target.")
		print("\nMany combinations of pre-processing exist and it can be lengthy to constantly go back and forth through your script to find optimal combinations of image pre-processors. This program should help you speed up your prototyping and quickly apply pre processing algorithms to images and see how they perform. You can then access the transformation log and get a code snippet that reflects the exact order of transformations applied to each specified working image.")
		print("\nFor more tips and information, visit the tips section in the <help> menu.")
		print("\n==========================")

		try:
		    self.pt_input("Press enter to continue", 'res')
		except SyntaxError:
		    pass

	def helpCommands(self):
		clear = lambda: os.system('clear')
		clear()
		print("=========HELP MENU========")
		print(" List of commands:")
		print("   - Important Commands")
		print("      - [load] - loads an image from the specified directory")
		print("      - [current], [curr] - displays information about the current working image")
		print("      - [retrieve], [use] - changes the current working image")
		print("      - [store], [save] - save an image state")
		print("      - [storeas], [saveas] - save an image state as a new name")
		print("      - [reset] - clear the working ledger")
		print("      - [showlist] - show the list of saved images")
		print("      - [drawimg], [draw] - draw the current working image")
		print("      - [q], [quit], [CTRL+C] - exit program")
		print("      - [help] - display the help menu")
		print("      - [genlog] - display the general log of actions")
		print("   - Manipulations")
		print("      - [log], [viewlog] - display the history of image transformations")
		print("      - [filt], [filter] - apply a general filter to the image")
		print("      - [conv] - Apply a convolution")
		print("      - [dog] - Apply a difference of gradients algorithm")
		print("      - [canny] - Apply the canny edge detection algorithm - Note: should be last step")
		print("==========================")

		try:
		    self.pt_input("Press enter to continue", 'res')
		except SyntaxError:
		    pass


	################################################################################
	################################################################################
	####																		####
	####								Main 			 					 	####
	####																		####
	################################################################################
	################################################################################
	def main(self):
		self.mainMessage()

		while self.go == True:
			try:
				option = self.pt_input(">", 'ias')
				continue

				# if(option == '0'):
				# 	confirm = self.pt_input(">>> Are you sure you want to quit?  Any unsaved progress will be lost (y/n)\n>")
				# 	if(confirm == 'y'):
				# 		print("===Goodbye===")
				# 		go == False
				# 		break
				# else:
				# 	print("... Command not recognized.  Type 'help' for detailed help")

					
			except Exception as e:
				print("=========ERROR=========")
				print(e)
				print("=======================")
				continue
			except KeyboardInterrupt as ki:
				print(ki)
				confirm = self.pt_input(">>> Are you sure? (y/n)\n", 'res')
				if confirm == 'y':
					break
		
program = Program()
program.setImageType("RGB")
program.setImagePath('/home/calvin/Documents/dod/training/')
program.main()


	
	