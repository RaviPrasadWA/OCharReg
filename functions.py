import numpy 
from pylab import mean

def area(a):
	return numpy.prod([max(x.stop-x.start,0) for x in a[:2]])

def dim0(s):
	return s[0].stop-s[0].start

def dim1(s):
	return s[1].stop-s[1].start

def xcenter(s):
	return mean([s[1].stop,s[1].start])

def ycenter(s):
	return mean([s[0].stop,s[0].start])

def center(s):
	return (ycenter(s),xcenter(s))

def width(s):
	return s[1].stop-s[1].start

def height(s):
	return s[0].stop-s[0].start

def aspect(a):
	return height(a)*1.0/width(a)