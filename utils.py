import PIL
import numpy
import re
import morph
import sys
import cPickle
import gzip
import unicodedata
import characters
import functions as sl
from decorators import checks
from types_ import 	ABINARY2,\
					SEGMENTATION,\
					PAGESEG,\
					AINT2,\
					AINT3,\
					GRAYSCALE,\
					ARANK,\
					LIGHTSEG,\
					DARKSEG
from pylab import 	amax,\
					amin,\
					unique,\
					array,\
					median,\
					zeros,\
					find,\
					inf,\
					ones,\
					clip,\
					where,\
					mean
from numpy import 	minimum, \
					dtype
from scipy.ndimage import 	measurements,\
							filters,\
							interpolation

class OpusException(Exception):
    trace = 1
    def __init__(self,*args,**kw):
        Exception.__init__(self,*args,**kw)

class record:
	def __init__(self,**kw): self.__dict__.update(kw)

def isintarray(a):
	return a.dtype in [dtype('B'),dtype('int16'),dtype('int32'),dtype('int64'),dtype('uint16'),dtype('uint32'),dtype('uint64')]

def isfloatarray(a):
	return a.dtype in [dtype('f'),dtype('float32'),dtype('float64')]

def isintegerarray(a):
	return a.dtype in [dtype('int32'),dtype('int64'),dtype('uint32'),dtype('uint64')]

def array2pil(a):
	if a.dtype==dtype("B"):
		if a.ndim==2:
			return PIL.Image.frombytes("L",(a.shape[1],a.shape[0]),a.tostring())
		elif a.ndim==3:
			return PIL.Image.frombytes("RGB",(a.shape[1],a.shape[0]),a.tostring())
		else:
			raise OpusException("bad image rank")
	elif a.dtype==dtype('float32'):
		return PIL.Image.fromstring("F",(a.shape[1],a.shape[0]),a.tostring())
	else:
		raise OpusException("unknown image type")

def midrange(	image,
				frac=0.5	):
	return frac*(amin(image)+amax(image))

def binary_objects(binary):
	labels,n 	= morph.label(binary)
	objects 	= morph.find_objects(labels)
	return objects

def compute_boxmap(	binary,
					scale,
					threshold=(.5,4),
					dtype='i'):
	objects 		= binary_objects(binary)
	bysize 			= sorted(objects,key=sl.area)
	boxmap 			= zeros(binary.shape,dtype)
	for o in bysize:
		if sl.area(o)**.5<threshold[0]*scale: 
			continue
		if sl.area(o)**.5>threshold[1]*scale: 
			continue
		boxmap[o] 	= 1
	return boxmap

def normalize_text(s):
	s 				= unicode(s)
	s 				= unicodedata.normalize('NFC',s)
	s 				= re.sub(ur'\s+(?u)',' ',s)
	s 				= re.sub(ur'\n(?u)','',s)
	s 				= re.sub(ur'^\s+(?u)','',s)
	s 				= re.sub(ur'\s+$(?u)','',s)
	for m,r in characters.replacements:
		s 			= re.sub(unicode(m),unicode(r),s)
	return s

def pil2array(	im,
				alpha=0	):
	if im.mode=="L":
		a = numpy.fromstring(im.tobytes(),'B')
		a.shape = im.size[1],im.size[0]
		return a
	if im.mode=="RGB":
		a = numpy.fromstring(im.tobytes(),'B')
		a.shape = im.size[1],im.size[0],3
		return a
	if im.mode=="RGBA":
		a = numpy.fromstring(im.tobytes(),'B')
		a.shape = im.size[1],im.size[0],4
		if not alpha: 
			a = a[:,:,:3]
		return a
	return pil2array(im.convert("L"))


@checks(str,_=ABINARY2)
def read_image_binary(	fname,
						dtype='i',
						pageno=0 	):
	if type(fname)==tuple: 
		fname,pageno = fname
	assert pageno==0
	pil 		= PIL.Image.open(fname)
	a 			= pil2array(pil)
	if a.ndim==3: 
		a 		= amax(a,axis=2)
	return array(a>0.5*(amin(a)+amax(a)),dtype)

#@checks(str,pageno=int,_=GRAYSCALE)
def read_image_gray(	pil,
						pageno=0	):
	a = pil2array(pil)
	if a.dtype==dtype('uint8'):
		a = a/255.0
	if a.dtype==dtype('int8'):
		a = a/127.0
	elif a.dtype==dtype('uint16'):
		a = a/65536.0
	elif a.dtype==dtype('int16'):
		a = a/32767.0
	elif isfloatarray(a):
		pass
	else:
		raise OpusException("unknown image type: "+a.dtype)
	if a.ndim==3:
		a = mean(a,2)
	return a

@checks(str)
def allsplitext(path):
	match = re.search(r'((.*/)*[^.]*)([^/]*)',path)
	if not match:
		return path,""
	else:
		return match.group(1),match.group(3)

def estimate_scale(binary):
	objects 		= binary_objects(binary)
	bysize 			= sorted(objects,key=sl.area)
	scalemap 		= zeros(binary.shape)
	for o in bysize:
		if amax(scalemap[o])>0: 
			continue
		scalemap[o] = sl.area(o)**0.5
	scale 			= median(scalemap[(scalemap>3)&(scalemap<100)])
	return scale

def compute_lines(	segmentation,
					scale 	):
	lobjects = morph.find_objects(segmentation)
	lines = []
	for i,o in enumerate(lobjects):
		if o is None: 
			continue
		if sl.dim1(o)<2*scale or sl.dim0(o)<scale: 
			continue
		mask 			= (segmentation[o]==i+1)
		if amax(mask)==0: 
			continue
		result 			= record()
		result.label 	= i+1
		result.bounds 	= o
		result.mask 	= mask
		lines.append(result)
	return lines

def reading_order(	lines,
					highlight=None,
					debug=0 	):
	order 			= zeros((len(lines),len(lines)),'B')
	def x_overlaps(u,v):
		return u[1].start<v[1].stop and u[1].stop>v[1].start
	def above(u,v):
		return u[0].start<v[0].start
	def left_of(u,v):
		return u[1].stop<v[1].start
	def separates(w,u,v):
		if w[0].stop<min(u[0].start,v[0].start): 
			return 0
		if w[0].start>max(u[0].stop,v[0].stop): 
			return 0
		if w[1].start<u[1].stop and w[1].stop>v[1].start: 
			return 1

	for i,u in enumerate(lines):
		for j,v in enumerate(lines):
			if x_overlaps(u,v):
				if above(u,v):
					order[i,j] = 1
			else:
				if [w for w in lines if separates(w,u,v)]==[]:
					if left_of(u,v): 
						order[i,j] = 1
			if j==highlight and order[i,j]:
				y0,x0 = sl.center(lines[i])
				y1,x1 = sl.center(lines[j])
	return order

def topsort(order):
	n = len(order)
	visited = zeros(n)
	L = []
	def visit(k):
		if visited[k]: 
			return
		visited[k] = 1
		for l in find(order[:,k]):
			visit(l)
		L.append(k)
	for k in range(n):
		visit(k)
	return L 

@checks(AINT2,_=AINT3)
def int2rgb(image):
	assert image.ndim==2
	assert isintarray(image)
	a 			= zeros(list(image.shape)+[3],'B')
	a[:,:,0] 	= (image>>16)
	a[:,:,1] 	= (image>>8)
	a[:,:,2] 	= image
	return a

@checks(DARKSEG,_=LIGHTSEG)
def make_seg_white(image):
	assert isintegerarray(image),"%s: wrong type for segmentation"%image.dtype
	image 			= image.copy()
	image[image==0] = 0xffffff
	return image

@checks(str,PAGESEG)
def write_page_segmentation(fname,image):
	assert image.ndim==2
	assert image.dtype in [dtype('int32'),dtype('int64')]
	a 	= int2rgb(make_seg_white(image))
	im 	= array2pil(a)
	im.save(fname)

def remove_noise(	line,
					minsize=8	):
	if minsize==0: 
		return line
	binary 		= (line>0.5*amax(line))
	labels,n 	= morph.label(binary)
	sums 		= measurements.sum(binary,labels,range(n+1))
	sums 		= sums[labels]
	good 		= minimum(binary,1-(sums>0)*(sums<minsize))
	return good

@checks(ARANK(2),int,int,int,int,mode=str,cval=True,_=GRAYSCALE)
def extract(	image,
				y0,
				x0,
				y1,
				x1,
				mode='nearest',
				cval=0	):
	h,w 					= image.shape
	ch,cw 					= y1-y0,x1-x0
	y,x 					= clip(y0,0,max(h-ch,0)),clip(x0,0,max(w-cw, 0))
	sub 					= image[y:y+ch,x:x+cw]
	try:
		r 					= interpolation.shift(sub,(y-y0,x-x0),mode=mode,cval=cval,order=0)
		if cw > w or ch > h:
			pady0, padx0 	= max(-y0, 0), max(-x0, 0)
			r 				= interpolation.affine_transform(r, eye(2), offset=(pady0, padx0), cval=1, output_shape=(ch, cw))
		return r
	except RuntimeError:
		dtype 				= sub.dtype
		sub 				= array(sub,dtype='float64')
		sub 				= interpolation.shift(sub,(y-y0,x-x0),mode=mode,cval=cval,order=0)
		sub 				= array(sub,dtype=dtype)
		return sub

def pad_image( 	image,
				d,
				cval=inf ):
    result 				= ones(array(image.shape)+2*d)
    result[:,:] 		= amax(image) if cval==inf else cval
    result[d:-d,d:-d] 	= image
    return result

@checks(ARANK(2),True,pad=int,expand=int,_=GRAYSCALE)
def extract_masked(	image,
					linedesc,
					pad=5,
					expand=0 ):
	y0,x0,y1,x1 = [	int(x) for x in [linedesc.bounds[0].start,linedesc.bounds[1].start,
					linedesc.bounds[0].stop,linedesc.bounds[1].stop]]
	if pad>0:
		mask 	= pad_image(linedesc.mask,pad,cval=0)
	else:
		mask 	= linedesc.mask
	line 		= extract(image,y0-pad,x0-pad,y1+pad,x1+pad)
	if expand>0:
		mask 	= filters.maximum_filter(mask,(expand,expand))
	line 		= where(mask,line,amax(line))
	return line

@checks(str,ABINARY2)
def write_image_binary(	fname,
						image,
						verbose=0	):
	assert image.ndim==2
	image 		= array(255*(image>midrange(image)),'B')
	im 			= array2pil(image)
	im.save(fname)


def write_image_gray(	fname,
						image,
						normalize=0,
						verbose=0	):
	if isfloatarray(image):
		image 	= array(255*clip(image,0.0,1.0),'B')
	assert image.dtype==dtype('B'),"array has wrong dtype: %s"%image.dtype
	im 			= array2pil(image)
	im.save(fname)

def unpickle_find_global(mname,cname):
	if mname == "ocrolib.lstm":
		mname	= mname.split(".")[-1]
	if mname == "ocrolib.lineest":
		mname	= "line_normalizer"
	if mname=="lstm.lstm":
		return getattr(lstm,cname)
	if not mname in sys.modules.keys():
		exec "import "+mname
	return getattr(sys.modules[mname],cname)

def load_object(fname,zip=0,nofind=0,verbose=0):
	with gzip.GzipFile(fname,"rb") as stream:
		unpickler = cPickle.Unpickler(stream)
		unpickler.find_global = unpickle_find_global
		return unpickler.load()
