import utils
import os
import functions as sl
from pylab import	amax,\
					where,\
					mean,\
					median,\
					isnan,\
					array,\
					minimum,\
					maximum,\
					zeros,\
					find
from decorators import checktype
from types_ import 	GRAYSCALE,\
					ABINARY2
from utils import	allsplitext,\
					read_image_binary,\
					estimate_scale,\
					compute_lines,\
					reading_order,\
					topsort,\
					remove_noise,\
					write_page_segmentation,\
					extract_masked,\
					write_image_gray,\
					write_image_binary,\
					read_image_gray
from scipy.ndimage import measurements
from scipy.ndimage.filters import	gaussian_filter, \
									uniform_filter,\
									maximum_filter
import morph

class Args:

	def __init__(self):
		self.sepwiden 		= 10.0
		self.maxseps 		= 0
		self.csminaspect 	= 1.1
		self.csminheight	= 10.0
		self.maxcolseps		= 3
		self.blackseps 		= True
		self.vscale			= 1.0
		self.hscale			= 1.0
		self.scale			= 0.0
		self.threshold		= 0.2
		self.nocheck		= False
		self.gray 			= True
		self.minscale		= 12.0
		self.maxlines		= 300
		self.usegauss		= True
		self.noise 			= 8
		self.pad 			= 3
		self.expand			= 3	

args 						= Args()

def norm_max(v):return v/amax(v)

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

	slots = int(w*h*1.0/(30*30))
	_,ncomps = measurements.label(image>mean(image))
	if ncomps<10: 
		return "too few connected components for a page image (got %d)"%(ncomps,)
	if ncomps>slots: 
		return "too many connnected components for a page image (%d > %d)"%(ncomps,slots)
	return None

def compute_separators_morph(	binary,
								scale 	):
	d0 			= int(max(5,scale/4))
	d1 			= int(max(5,scale))+args.sepwiden
	thick 		= morph.r_dilation(binary,(d0,d1))
	vert 		= morph.rb_opening(thick,(10*scale,1))
	vert 		= morph.r_erosion(vert,(d0//2,args.sepwiden))
	vert 		= morph.select_regions(vert,sl.dim1,min=3,nbest=2*args.maxseps)
	vert 		= morph.select_regions(vert,sl.dim0,min=20*scale,nbest=args.maxseps)
	return vert

def compute_colseps_morph(	binary,
							scale,
							maxseps=3,
							minheight=20,
							maxwidth=5 	):
	boxmap 		= utils.compute_boxmap(binary,scale,dtype='B')
	bounds 		= morph.rb_closing(B(boxmap),(int(5*scale),int(5*scale)))
	bounds 		= maximum(B(1-bounds),B(boxmap))
	cols 		= 1-morph.rb_closing(boxmap,(int(20*scale),int(scale)))
	cols 		= morph.select_regions(cols,sl.aspect,min=args.csminaspect)
	cols 		= morph.select_regions(cols,sl.dim0,min=args.csminheight*scale,nbest=args.maxcolseps)
	cols 		= morph.r_erosion(cols,(int(0.5+scale),0))
	cols 		= morph.r_dilation(cols,(int(0.5+scale),0),origin=(int(scale/2)-1,0))
	return cols

def compute_colseps_mconv(	binary,
							scale=1.0 	):
	h,w 		= binary.shape
	smoothed 	= gaussian_filter(1.0*binary,(scale,scale*0.5))
	smoothed 	= uniform_filter(smoothed,(5.0*scale,1))
	thresh 		= (smoothed<amax(smoothed)*0.1)
	blocks 		= morph.rb_closing(binary,(int(4*scale),int(4*scale)))
	seps 		= minimum(blocks,thresh)
	seps 		= morph.select_regions(seps,sl.dim0,min=args.csminheight*scale,nbest=args.maxcolseps)
	blocks 		= morph.r_dilation(blocks,(5,5))
	seps 		= maximum(seps,1-blocks)
	return seps

def compute_colseps_conv( 	binary,
							scale=1.0	):
	h,w 		= binary.shape
	smoothed 	= gaussian_filter(1.0*binary,(scale,scale*0.5))
	smoothed 	= uniform_filter(smoothed,(5.0*scale,1))
	thresh 		= (smoothed<amax(smoothed)*0.1)
	grad 		= gaussian_filter(1.0*binary,(scale,scale*0.5),order=(0,1))
	grad 		= uniform_filter(grad,(10.0*scale,1))
	grad 		= (grad>0.5*amax(grad))
	seps 		= minimum(thresh,maximum_filter(grad,(int(scale),int(5*scale))))
	seps 		= maximum_filter(seps,(int(2*scale),1))
	seps 		= morph.select_regions(seps,sl.dim0,min=args.csminheight*scale,nbest=args.maxcolseps)
	return seps

def compute_colseps( 	binary,
						scale 	):
    colseps 	= compute_colseps_conv(binary,scale)
    if args.blackseps and args.maxseps == 0:
    	args.maxseps = 2
    if args.maxseps > 0:
    	seps 	= compute_separators_morph(binary,scale)
    	colseps = maximum(colseps,seps)
    	binary 	= minimum(binary,1-seps)
    return colseps,binary

def compute_gradmaps(	binary,
						scale 	):
	boxmap 		= utils.compute_boxmap(binary,scale)
	cleaned 	= boxmap*binary
	if args.usegauss:
		grad 	= gaussian_filter(	1.0*cleaned,
									(args.vscale*0.3*scale,args.hscale*6*scale),
									order=(1,0))
	else:
		grad 	= gaussian_filter(	1.0*cleaned,
									(max(4,args.vscale*0.3*scale),args.hscale*scale),
									order=(1,0))
		grad 	= uniform_filter(grad,(args.vscale,args.hscale*6*scale))
	bottom 		= norm_max((grad<0)*(-grad))
	top 		= norm_max((grad>0)*grad)
	return bottom,top,boxmap

def compute_line_seeds(	binary,
						bottom,
						top,
						colseps,
						scale ):
	t 			= args.threshold
	vrange 		= int(args.vscale*scale)
	bmarked 	= maximum_filter(bottom==maximum_filter(bottom,(vrange,0)),(2,2))
	bmarked 	= bmarked*(bottom>t*amax(bottom)*t)*(1-colseps)
	tmarked 	= maximum_filter(top==maximum_filter(top,(vrange,0)),(2,2))
	tmarked 	= tmarked*(top>t*amax(top)*t/2)*(1-colseps)
	tmarked 	= maximum_filter(tmarked,(1,20))
	seeds 		= zeros(binary.shape,'i')
	delta = max(3,int(scale/2))
	for x in range(bmarked.shape[1]):
		transitions = sorted([(y,1) for y in find(bmarked[:,x])]+[(y,0) for y in find(tmarked[:,x])])[::-1]
		transitions += [(0,0)]
		for l in range(len(transitions)-1):
			y0,s0 	= transitions[l]
			if s0==0: 
				continue
			seeds[y0-delta:y0,x] = 1
			y1,s1 	= transitions[l+1]
			if s1==0 and (y0-y1)<5*scale: 
				seeds[y1:y0,x] = 1
	seeds 		= maximum_filter(seeds,(1,int(1+scale)))
	seeds 		= seeds*(1-colseps)
	seeds,_ 	= morph.label(seeds)
	return seeds

def remove_hlines( 	binary,	
					scale,
					maxsize=10 	):
	labels,_ 	= morph.label(binary)
	objects 	= morph.find_objects(labels)
	for i,b in enumerate(objects):
		if sl.width(b)>maxsize*scale:
			labels[b][labels[b]==i+1] = 0
	return array(labels!=0,'B')

def compute_segmentation(	binary,
							scale	):
	binary 				= array(binary,'B')
	binary 				= remove_hlines(binary,scale)
	colseps,binary 		= compute_colseps(binary,scale)
	bottom,top,boxmap 	= compute_gradmaps(binary,scale)
	seeds 				= compute_line_seeds(binary,bottom,top,colseps,scale)
	llabels 			= morph.propagate_labels(boxmap,seeds,conflict=0)
	spread 				= morph.spread_labels(seeds,maxdist=scale)
	llabels 			= where(llabels>0,llabels,spread*binary)
	segmentation 		= llabels*binary
	return segmentation

def segementation_run(input_):
	binary,gray 	= input_
	checktype(binary,ABINARY2)
	if not args.nocheck:
		check 		= check_page(amax(binary)-binary)
		if check is not None:
			return
	if args.gray:
		checktype(gray,GRAYSCALE)
	binary 			= 1-binary
	if args.scale==0:
		scale 		= estimate_scale(binary)
	else:
		scale 		= args.scale

	if isnan(scale) or scale>1000.0:
		return
	if scale<args.minscale:
		return
	segmentation 	= compute_segmentation(binary,scale)
	if amax(segmentation)>args.maxlines:
		return

	lines 			= compute_lines(segmentation,scale)
	order 			= reading_order([l.bounds for l in lines])
	lsort 			= topsort(order)
	nlabels 		= amax(segmentation)+1
	renumber = zeros(nlabels,'i')
	for i,v in enumerate(lsort): 
		renumber[lines[v].label] = 0x010000+(i+1)
	segmentation 	= renumber[segmentation]
	lines 			= [lines[i] for i in lsort]
	#write_page_segmentation("%s.pseg.png"%outputdir,segmentation)
	cleaned 		= remove_noise(binary,args.noise)
	for i,l in enumerate(lines):
		binline 	= extract_masked(1-cleaned,l,pad=args.pad,expand=args.expand)
		#write_image_binary("%s/01%04x.bin.png"%(outputdir,i+1),binline)
		if args.gray:
			grayline = extract_masked(gray,l,pad=args.pad,expand=args.expand)
			#write_image_gray("%s/01%04x.nrm.png"%(outputdir,i+1),grayline)
			yield (binline,grayline)
