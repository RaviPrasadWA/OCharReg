from scipy import stats
from utils import 	read_image_gray,\
					allsplitext,\
					write_image_binary,\
					write_image_gray
from scipy.ndimage import 	filters,\
							interpolation,\
							morphology
from pylab import	amin,\
					amax,\
					sum,\
					prod,\
					minimum,\
					clip,\
					array,\
					mean,\
					median,\
					linspace,\
					var,\
					ones


class Args:
	def __init__(self):
		self.nocheck	= True
		self.gray 		= True
		self.maxskew 	= 2.0
		self.threshold	= 0.5
		self.zoom		= 0.5
		self.escale 	= 1.0
		self.bignore	= 0.1
		self.perc 		= 80.0
		self.range 		= 20
		self.maxskew	= 2.0
		self.lo			= 5.0
		self.hi			= 90.0
		self.skewsteps 	= 8

args 					= Args()

def estimate_skew_angle(	image,
							angles	):
	estimates = []
	for a in angles:
		v 				= mean(interpolation.rotate(image,a,order=0,mode='constant'),axis=1)
		v 				= var(v)
		estimates.append((v,a))
	_,a 				= max(estimates)
	return a

def check_page(image):
	if len(image.shape)==3: 
		return "input image is color image %s"%(image.shape,)
	if mean(image)<median(image): 
		return "image may be inverted"
	h,w = image.shape
	if h<600: 
		return "image not tall enough for a page image %s"%(image.shape,)
	if h>10000: 
		return "image too tall for a page image %s"%(image.shape,)
	if w<600: 
		return "image too narrow for a page image %s"%(image.shape,)
	if w>10000: 
		return "line too wide for a page image %s"%(image.shape,)
	return None

def binarize_image(job):
	image_object,i 		= job
	raw 				= read_image_gray(image_object)
	image 				= raw - amin(raw)
	if amax(image)==amin(image):
		return # Image is empty
	image 				/= amax(image)
	check 				= check_page(amax(image)-image)
	if check is not None:
		return
	if args.gray:
		extreme 		= 0
	else:
		extreme 		= (sum(image<0.05)+sum(image>0.95))*1.0/prod(image.shape)

	if extreme>0.95:
		comment = "no-normalization"
		flat = image
	else:
		comment 		= ""
		m 				= interpolation.zoom(image,args.zoom)
		m 				= filters.percentile_filter(m,args.perc,size=(args.range,2))
		m 				= filters.percentile_filter(m,args.perc,size=(2,args.range))
		m 				= interpolation.zoom(m,1.0/args.zoom)
		w,h 			= minimum(array(image.shape),array(m.shape))
		flat 			= clip(image[:w,:h]-m[:w,:h]+1,0,1)

	if args.maxskew>0:
		d0,d1 			= flat.shape
		o0,o1 			= int(args.bignore*d0),int(args.bignore*d1)
		flat 			= amax(flat)-flat
		flat 			-= amin(flat)
		est 			= flat[o0:d0-o0,o1:d1-o1]
		ma 				= args.maxskew
		ms 				= int(2*args.maxskew*args.skewsteps)
		angle 			= estimate_skew_angle(est,linspace(-ma,ma,ms+1))
		flat 			= interpolation.rotate(flat,angle,mode='constant',reshape=0)
		flat 			= amax(flat)-flat
	else:
		angle 			= 0

	d0,d1 				= flat.shape
	o0,o1 				= int(args.bignore*d0),int(args.bignore*d1)
	est 				= flat[o0:d0-o0,o1:d1-o1]

	if args.escale>0:
		e 				= args.escale
		v 				= est-filters.gaussian_filter(est,e*20.0)
		v 				= filters.gaussian_filter(v**2,e*20.0)**0.5
		v 				= (v>0.3*amax(v))
		v 				= morphology.binary_dilation(v,structure=ones((int(e*50),1)))
		v 				= morphology.binary_dilation(v,structure=ones((1,int(e*50))))
		est 			= est[v]
	lo 					= stats.scoreatpercentile(est.ravel(),args.lo)
	hi 					= stats.scoreatpercentile(est.ravel(),args.hi)
	flat 				-= lo
	flat 				/= (hi-lo)
	flat 				= clip(flat,0,1)
	binary 				= 1*(flat>args.threshold)
	return (binary,flat)

