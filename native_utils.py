import contextlib
import os
import hashlib
import ctypes
from numpy.ctypeslib import ndpointer
from ctypes import 	c_int,\
					c_float,\
					c_double,\
					c_byte

I 							= c_int
F 							= c_float
D 							= c_double
B 							= c_byte

@contextlib.contextmanager
def lockfile(fname,delay=0.5):
	while 1:
		try:
			fd 				= os.open(fname,os.O_RDWR|os.O_CREAT|os.O_EXCL)
		except OSError as e:
			if e.errno!=errno.EEXIST: raise
			time.sleep(delay)
			continue
		else:
			break
	try:
		yield fd
	finally:
		os.close(fd)
		os.unlink(fname)

for d in range(1,4):
	for T,t in [("I","int32"),("F","float32"),("D","float64"),("B","int8"),("U","uint8")]:
		exec "A%d%s = ndpointer(dtype='%s',ndim=%d,flags='CONTIGUOUS,ALIGNED')"%(d,T,t,d)

class CompileError(Exception):
	pass

def compile_and_find(	c_string,prefix=".pynative",
						opt="-g -O4",
						libs="-lm",
						options="-shared -fopenmp -std=c99 -fPIC",
						verbose=0	):
	if not os.path.exists(prefix):
		os.mkdir(prefix)
	m 						= hashlib.md5()
	m.update(c_string)
	base 					= m.hexdigest()
	with lockfile(os.path.join(prefix,base+".lock")):
		so 					= os.path.join(prefix,base+".so")
		if os.path.exists(so):
			return so
		source 				= os.path.join(prefix,base+".c")
		with open(source,"w") as stream:
			stream.write(c_string)
		cmd 				= "gcc "+opt+" "+libs+" "+options+" "+source+" -o "+so
		if os.system(cmd)!=0:
			raise CompileError()
		return so

def compile_and_load(c_string,**keys):
	path 					= compile_and_find(c_string,**keys)
	return ctypes.CDLL(path)

lstm_utils = r"""
	#include <math.h>

	void sumouter(int r,int n,int m,double out[n][m],double u[r][n],double v[r][m]) {
	    for(int i=0;i<n;i++) {
	        for(int j=0;j<m;j++) {
	            double total = 0.0;
	            for(int k=0;k<r;k++) total += u[k][i]*v[k][j];
	            out[i][j] = total;
	        }
	    }
	}

	void sumprod(int r,int n,double out[n],double u[r][n],double v[r][n]) {
	    for(int i=0;i<n;i++) {
	        double total = 0.0;
	        for(int k=0;k<r;k++) total += u[k][i]*v[k][i];
	        out[i] = total;
	    }
	}
"""

lstm_native 					= compile_and_load(lstm_utils)
lstm_native.sumouter.argtypes 	= [I,I,I,A2D,A2D,A2D]
lstm_native.sumprod.argtypes 	= [I,I,A1D,A2D,A2D]

def sumouter(u,v,out=None):
	assert out.shape==u.shape[1:]+v.shape[1:] and u.shape[:1]==v.shape[:1]
	lstm_native.sumouter(u.shape[0],out.shape[0],out.shape[1],out,u,v)
	return out

def sumprod(u,v,out=None):
	assert out.shape==u.shape[1:] and out.shape==v.shape[1:] and u.shape[:1]==v.shape[:1]
	lstm_native.sumprod(len(u),len(out),out,u,v)
	return out