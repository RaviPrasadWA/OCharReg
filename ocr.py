from binarize import binarize_image
from segment import segementation_run
from predict import predict
from collections import OrderedDict
from PIL import Image
import json

TEST = False

def optical_character_recognize(image_object):
	index 					= 0
	text_data				= OrderedDict()
	binary_image			= binarize_image((image_object,0))
	if binary_image:
		segments					= segementation_run(binary_image)
		if segments:
			for binline,grayline in segments:
				text_data[index] 	= predict(binline)
				index				+= 1
	return json.dumps(text_data)

if TEST:
	f = Image.open('examples/img1.png')
	print optical_character_recognize(f)