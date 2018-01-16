from scipy.ndimage import 	morphology,\
							measurements,\
							filters
from scipy.ndimage.morphology import *
from pylab import 	argsort,\
					zeros,\
					array,\
					amin,\
					amax,\
					unique
from types_ import 	ABINARY2, \
					AINT2,\
					uintpair,\
					SEGMENTATION
from decorators import checks

@checks(SEGMENTATION,SEGMENTATION)
def correspondences(	labels1,
						labels2	):
	q 		= 100000
	assert amin(labels1)>=0 and amin(labels2)>=0
	assert amax(labels2)<q
	combo 	= labels1*q+labels2
	result 	= unique(combo)
	result 	= array([result//q,result%q])
	return result

@checks(ABINARY2)
def label(image,**kw):
	try: 
		return measurements.label(image,**kw)
	except: 
		pass
	types = [	"int32",
				"uint32",
				"int64",
				"unit64",
				"int16",
				"uint16" 	]
	for t in types:
		try: 
			return measurements.label(array(image,dtype=t),**kw)
		except: 
			pass
	return measurements.label(image,**kw)

@checks(AINT2)
def find_objects(	image,
					**kw 	):
	try: 
		return measurements.find_objects(image,**kw)
	except: 
		pass
	types = [	"int32",
				"uint32",
				"int64",
				"unit64",
				"int16",
				"uint16"	]
	for t in types:
		try: 
			return measurements.find_objects(array(image,dtype=t),**kw)
		except: 
			pass
	return measurements.find_objects(image,**kw)

@checks(ABINARY2,True)
def select_regions(	binary,
					f,
					min=0,
					nbest=100000 ):
	labels,n 	= label(binary)
	objects 	= find_objects(labels)
	scores 		= [f(o) for o in objects]
	best 		= argsort(scores)
	keep 		= zeros(len(objects)+1,'i')
	if nbest > 0:
		for i in best[-nbest:]:
			if scores[i]<=min: 
				continue
			keep[i+1] = 1
	return keep[labels]

@checks(ABINARY2,uintpair)
def r_dilation(	image,
				size,
				origin=0 ):
	return filters.maximum_filter(image,size,origin=origin)

@checks(ABINARY2,uintpair)
def r_erosion(	image,
				size,
				origin=0	):
	return filters.minimum_filter(image,size,origin=origin)

@checks(ABINARY2,uintpair)
def rb_dilation(	image,
					size,
					origin=0	):
	output 					= zeros(image.shape,'f')
	filters.uniform_filter(	image,
							size,
							output=output,
							origin=origin,
							mode='constant',
							cval=0	)
	return array(output>0,'i')

@checks(ABINARY2,uintpair)
def rb_erosion(	image,
				size,
				origin=0	):
	output 					= zeros(image.shape,'f')
	filters.uniform_filter(	image,
							size,
							output=output,
							origin=origin,
							mode='constant',
							cval=1	)
	return array(output==1,'i')

@checks(ABINARY2,uintpair)
def rb_opening(	image,
				size,
				origin=0	):
	image 					= rb_erosion(image,size,origin=origin)
	return rb_dilation(image,size,origin=origin)

@checks(ABINARY2,uintpair)
def rb_closing(	image,
				size,
				origin=0	):
	image = rb_dilation(image,size,origin=origin)
	return rb_erosion(image,size,origin=origin)

@checks(ABINARY2,SEGMENTATION)
def propagate_labels(	image,
						labels,
						conflict=0	):
	rlabels,_ 				= label(image)
	cors 					= correspondences(rlabels,labels)
	outputs 				= zeros(amax(rlabels)+1,'i')
	oops 					= -(1<<30)
	for o,i in cors.T:
		if outputs[o]!=0: 
			outputs[o] 		= oops
		else: 
			outputs[o] 		= i
	outputs[outputs==oops] 	= conflict
	outputs[0] 				= 0
	return outputs[rlabels]

@checks(SEGMENTATION)
def spread_labels(	labels,
					maxdist	=9999999	):
	distances,features 		= morphology.distance_transform_edt(labels==0,
																return_distances=1,
																return_indices=1)
	indexes 				= features[0]*labels.shape[1]+features[1]
	spread 					= labels.ravel()[indexes.ravel()].reshape(*labels.shape)
	spread 					*= (distances<maxdist)
	return spread
