from decorators import makeargcheck
import numpy

int_dtypes 			= [	numpy.dtype('uint8'),
						numpy.dtype('int32'),
						numpy.dtype('int64'),
						numpy.dtype('uint32'),
						numpy.dtype('uint64')	]
float_dtypes 		= [ numpy.dtype('float32'),
						numpy.dtype('float64')	]
try: 
	float_dtypes 	+= [numpy.dtype('float96')]
except: 
	pass
try: 
	float_dtypes 	+= [numpy.dtype('float128')]
except: 
	pass

@makeargcheck("array must contain floating point values")
def AFLOAT(a):
	return a.dtype in float_dtypes

@makeargcheck("expect pair of int")
def uintpair(a):
	if not tuple(a): 
		return 0
	if not len(a)==2: 
		return 0
	if a[0]<0: 
		return 0
	if a[1]<0: 
		return 0
	return 1

def ALL(*checks):
	def CHK_(x):
		for check in checks:
			check(x)
	return CHK_

@makeargcheck("array must contain integer values")
def AINT(a):
	return a.dtype in int_dtypes

def ARANK(n):
	@makeargcheck("array must have rank %d"%n)
	def ARANK_(a):
		if not hasattr(a,"ndim"): 
			return 0
		return a.ndim==n
	return ARANK_

@makeargcheck("expected a boolean array or an array of 0/1")
def ABINARY(a):
	if a.ndim==2 and a.dtype==numpy.dtype(bool): 
		return 1
	if not a.dtype in int_dtypes: 
		return 0

	import scipy.ndimage.measurements
	zeros,ones = scipy.ndimage.measurements.sum(1,a,[0,1])
	if zeros+ones == a.size: 
		return 1
	if a.dtype==numpy.dtype('B'):
		zeros,ones = scipy.ndimage.measurements.sum(1,a,[0,255])
		if zeros+ones == a.size: 
			return 1
	return 0

@makeargcheck("expected a segmentation image")
def SEGMENTATION(a):
	return isinstance(a,numpy.ndarray) and a.ndim==2 and a.dtype in ['int32','int64']

@makeargcheck("expected a segmentation with black background")
def BLACKSEG(a):
	return numpy.amax(a)<0xffffff

@makeargcheck("expect a page image (larger than 600x600)",warning=1)
def PAGE(a):
	return a.ndim==2 and a.shape[0]>=600 and a.shape[1]>=600

@makeargcheck("all non-zero pixels in a page segmentation must have a column value >0")
def PAGEEXTRA(a):
	u 		= numpy.unique(a)
	u 		= u[u!=0]
	u 		= u[(u&0xff0000)==0]
	return len(u)==0

@makeargcheck("expected a segmentation with white background")
def WHITESEG(a):
	return numpy.amax(a)==0xffffff

@makeargcheck("expected a segmentation with black background")
def BLACKSEG(a):
	return numpy.amax(a)<0xffffff

ARRAY2 		= ARANK(2)
ABINARY2 	= ALL(ABINARY,ARRAY2)
AINT2 		= ALL(ARANK(2),AINT)
AINT3 		= ALL(ARANK(3),AINT)
AFLOAT2 	= ALL(ARANK(2),AFLOAT)
GRAYSCALE 	= AFLOAT2
PAGESEG 	= ALL(SEGMENTATION,BLACKSEG,PAGE,PAGEEXTRA)
LIGHTSEG 	= ALL(SEGMENTATION,WHITESEG)
DARKSEG 	= ALL(SEGMENTATION,BLACKSEG)