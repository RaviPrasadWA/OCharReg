from lstm_functions import  forward_py,\
                            backward_py,\
                            initial_range,\
                            normalize_nfkc,\
                            prepare_line
from pylab import   array,\
                    concatenate,\
                    dot,\
                    array,\
                    nan,\
                    ones,\
                    isnan
from neural_network_base import Network
from logreg import Logreg
from softmax import Softmax
from reversed_network import Reversed
from parallel_network import Parallel
from stacked_network import Stacked
from mlp import MLP
from codec import Codec

class RangeError(Exception):
    def __init__(   self,
                    s=None  ):
        Exception.__init__(self,s)

class OpusException(Exception):
    trace                   = 1
    def __init__(self,*args,**kw):
        Exception.__init__(self,*args,**kw)

class RecognitionError(OpusException):
    trace                   = 1
    def __init__(self,explanation,**kw):
        self.context        = kw
        s                   = [explanation]
        s                   += ["%s=%s"%(k,summary(kw[k])) for k in kw]
        message             = " ".join(s)
        Exception.__init__(self,message)

class LSTM(Network):

    def __init__(self,ni,ns,initial=initial_range,maxlen=5000):
        na                  = 1+ni+ns
        self.dims           = ni,ns,na
        self.init_weights(initial)
        self.allocate(maxlen)

    def ninputs(self):
        return self.dims[0]

    def noutputs(self):
        return self.dims[1]

    def states(self):
        return array(self.state[:self.last_n])

    def init_weights(self,initial):
        ni,ns,na            = self.dims
        for w in "WGI WGF WGO WCI".split():
            setattr(self,w,randu(ns,na)*initial)
            setattr(self,"D"+w,zeros((ns,na)))
        for w in "WIP WFP WOP".split():
            setattr(self,w,randu(ns)*initial)
            setattr(self,"D"+w,zeros(ns))

    def weights(self):
        weights             = "WGI WGF WGO WCI WIP WFP WOP"
        for w in weights.split():
            yield(getattr(self,w),getattr(self,"D"+w),w)
    
    def info(self):
        vars_               = "WGI WGF WGO WIP WFP WOP cix ci gix gi gox go gfx gf"
        vars_               += " source state output gierr gferr goerr cierr stateerr"
        vars_               = vars_.split()
        vars_               = sorted(vars_)
        for v in vars_:
            a               = array(getattr(self,v))

    def preSave(self):
        self.max_n          = max(500,len(self.ci))
        self.allocate(1)

    def postLoad(self):
        self.allocate(getattr(self,"max_n",5000))

    def allocate(self,n):
        ni,ns,na            = self.dims
        vars_               = "cix ci gix gi gox go gfx gf"
        vars_               += " state output gierr gferr goerr cierr stateerr outerr"
        for v in vars_.split():
            setattr(self,v,nan*ones((n,ns)))
        self.source         = nan*ones((n,na))
        self.sourceerr      = nan*ones((n,na))

    def reset(self,n):
        vars_                       = "cix ci gix gi gox go gfx gf"
        vars_                       += " state output gierr gferr goerr cierr stateerr outerr"
        vars_                       += " source sourceerr"
        for v in vars_.split():
            getattr(self,v)[:,:]    = nan

    def forward(self,xs):
        ni,ns,na                    = self.dims
        assert len(xs[0])==ni
        n                           = len(xs)
        self.last_n                 = n
        N                           = len(self.gi)
        if n>N: 
            raise RecognitionError("[i] Input too large for model")
        self.reset(n)
        forward_py( n,N,ni,ns,na,xs,
                   self.source,
                   self.gix,self.gfx,self.gox,self.cix,
                   self.gi,self.gf,self.go,self.ci,
                   self.state,self.output,
                   self.WGI,self.WGF,self.WGO,self.WCI,
                   self.WIP,self.WFP,self.WOP)
        assert not isnan(self.output[:n]).any()
        return self.output[:n]

    def backward(self,deltas):
        ni,ns,na                    = self.dims
        n                           = len(deltas)
        self.last_n                 = n
        N                           = len(self.gi)
        if n>N: 
            raise ocrolib.RecognitionError("[i] Input too large for model")
        backward_py(n,N,ni,ns,na,deltas,
                    self.source,
                    self.gix,self.gfx,self.gox,self.cix,
                    self.gi,self.gf,self.go,self.ci,
                    self.state,self.output,
                    self.WGI,self.WGF,self.WGO,self.WCI,
                    self.WIP,self.WFP,self.WOP,
                    self.sourceerr,
                    self.gierr,self.gferr,self.goerr,self.cierr,
                    self.stateerr,self.outerr,
                    self.DWGI,self.DWGF,self.DWGO,self.DWCI,
                    self.DWIP,self.DWFP,self.DWOP)
        return [s[1:1+ni] for s in self.sourceerr[:n]]


def BIDILSTM(Ni,Ns,No):
    lstm1           = LSTM(Ni,Ns)
    lstm2           = Reversed(LSTM(Ni,Ns))
    bidi            = Parallel(lstm1,lstm2)
    assert No>1
    logreg          = Softmax(2*Ns,No)
    stacked         = Stacked([bidi,logreg])
    return stacked

from sequence import SeqRecognizer
