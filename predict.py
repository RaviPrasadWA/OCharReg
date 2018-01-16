from lstm import LSTM
from scipy.ndimage import measurements
from utils import 	load_object,\
					allsplitext,\
					read_image_gray,\
					normalize_text
from pylab import 	prod,\
					amin,\
					amax,\
					mean,\
					median
import lstm

class Args:
	def __init__(self):
		self.model		= "model/en.gz"
		self.height 	= -1
		self.nocheck 	= False
		self.nolineest 	= False
		self.nonormalize= True
		self.pad 		= 16

args 					= Args()
network 				= load_object(args.model,verbose=1)

for x in network.walk(): 
	x.postLoad()
for x in network.walk():
	if isinstance(x,LSTM):
		x.allocate(5000)

lnorm 					= getattr(network,"lnorm",None)
if args.height>0:
	lnorm.setHeight(args.height)

def check_line(image):
	if len(image.shape)==3: 
		return "input image is color image %s"%(image.shape,)
	if mean(image)<median(image): 
		return "image may be inverted"
	h,w 				= image.shape
	if h<20:
		return "image not tall enough for a text line %s"%(image.shape,)
	if h>200:
		return "image too tall for a text line %s"%(image.shape,)
	if w<1.5*h:
		return "line too short %s"%(image.shape,)
	if w>4000:
		return "line too long %s"%(image.shape,)
	ratio 				= w*1.0/h
	_,ncomps 			= measurements.label(image>mean(image))
	lo 					= int(0.5*ratio+0.5)
	hi 					= int(4*ratio)+1
	if ncomps<lo:
		return "too few connected components (got %d, wanted >=%d)"%(ncomps,lo)
	if ncomps>hi*ratio:
		return "too many connected components (got %d, wanted <=%d)"%(ncomps,hi)
	return None


def predict(line):
	raw_line 			= line.copy()
	if prod(line.shape)==0: 
		return None
	if amax(line)==amin(line): 
		return None

	if not args.nocheck:
		check 			= check_line(amax(line)-line)
		if check is not None:
			return None

	if not args.nolineest:
		temp 			= amax(line)-line
		temp 			= temp*1.0/amax(temp)
		lnorm.measure(temp)
		line 			= lnorm.normalize(line,cval=amax(line))
	else:
		assert "dew.png" in fname,"only apply to dewarped images"

	line 				= lstm.prepare_line(line,args.pad)
	pred 				= network.predictString(line)

	if not args.nonormalize:
		pred 			= normalize_text(pred)
	return pred
