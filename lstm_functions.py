import unicodedata
from pylab import   vstack,\
                    amax,\
                    zeros,\
                    sum,\
                    exp,\
                    rand,\
                    isnan,\
                    outer,\
                    clip,\
                    tanh,\
                    dot,\
                    isnan,\
                    argmax,\
                    log,\
                    where,\
                    maximum,\
                    tile,\
                    roll,\
                    array,\
                    arange
import native_utils as nutils
from scipy.ndimage import measurements

initial_range = 0.1

def prepare_line(   line,
                    pad=16  ):
    line        = line * 1.0/amax(line)
    line        = amax(line)-line
    line        = line.T
    if pad>0:
        w       = line.shape[1]
        line    = vstack([zeros((pad,w)),line,zeros((pad,w))])
    return line

def randu(*shape):
    return 2*rand(*shape)-1

def sigmoid(x):
    return 1.0/(1.0+exp(-x))

def rownorm(a):
    return sum(array(a)**2,axis=1)**.5

def check_nan(  *args,
                **kw    ):
    for arg in args:
        if isnan(arg).any():
            raise FloatingPointError()

def sumouter(us,vs,lo=-1.0,hi=1.0,out=None):
    result          = out or zeros((len(us[0]),len(vs[0])))
    for u,v in zip(us,vs):
        result      += outer(clip(u,lo,hi),v)
    return result

def sumprod(us,vs,lo=-1.0,hi=1.0,out=None):
    assert len(us[0])==len(vs[0])
    result          = out or zeros(len(us[0]))
    for u,v in zip(us,vs):
        result      += clip(u,lo,hi)*v
    return result

def ffunc(x):
    return 1.0/(1.0+exp(-x))

def fprime(x,y=None):
    if y is None: 
        y           = sigmoid(x)
    return y*(1.0-y)

def gfunc(x):
    return tanh(x)

def gprime(x,y=None):
    if y is None: 
        y           = tanh(x)
    return 1-y**2

def hfunc(x):
    return tanh(x)

def hprime(x,y=None):
    if y is None: 
        y           = tanh(x)
    return 1-y**2


def forward_py(n,N,ni,ns,na,xs,source,gix,gfx,gox,cix,gi,gf,go,ci,state,output,WGI,WGF,WGO,WCI,WIP,WFP,WOP):
    for t in range(n):
        prev                    = zeros(ns) if t==0 else output[t-1]
        source[t,0]             = 1
        source[t,1:1+ni]        = xs[t]
        source[t,1+ni:]         = prev
        dot(WGI,source[t],out=gix[t])
        dot(WGF,source[t],out=gfx[t])
        dot(WGO,source[t],out=gox[t])
        dot(WCI,source[t],out=cix[t])
        if t>0:
            gix[t]              += WIP*state[t-1]
            gfx[t]              += WFP*state[t-1]
        gi[t]                   = ffunc(gix[t])
        gf[t]                   = ffunc(gfx[t])
        ci[t]                   = gfunc(cix[t])
        state[t]                = ci[t]*gi[t]
        if t>0:
            state[t]            += gf[t]*state[t-1]
            gox[t]              += WOP*state[t]
        go[t]                   = ffunc(gox[t])
        output[t]               = hfunc(state[t]) * go[t]
    assert not isnan(output[:n]).any()

def backward_py(n,N,ni,ns,na,deltas,source,gix,gfx,gox,cix,gi,gf,go,ci,state,output,WGI,WGF,WGO,WCI,WIP,WFP,WOP,sourceerr,gierr,gferr,goerr,cierr,stateerr,outerr,DWGI,DWGF,DWGO,DWCI,DWIP,DWFP,DWOP):
    for t in reversed(range(n)):
        outerr[t]               = deltas[t]
        if t<n-1:
            outerr[t]           += sourceerr[t+1][-ns:]
        goerr[t]                = fprime(None,go[t]) * hfunc(state[t]) * outerr[t]
        stateerr[t]             = hprime(state[t]) * go[t] * outerr[t]
        stateerr[t]             += goerr[t]*WOP
        if t<n-1:
            stateerr[t]         += gferr[t+1]*WFP
            stateerr[t]         += gierr[t+1]*WIP
            stateerr[t]         += stateerr[t+1]*gf[t+1]
        if t>0:
            gferr[t]            = fprime(None,gf[t])*stateerr[t]*state[t-1]
        gierr[t]                = fprime(None,gi[t])*stateerr[t]*ci[t] # gfunc(cix[t])
        cierr[t]                = gprime(None,ci[t])*stateerr[t]*gi[t]
        dot(gierr[t],WGI,out=sourceerr[t])
        if t>0:
            sourceerr[t]        += dot(gferr[t],WGF)
        sourceerr[t]            += dot(goerr[t],WGO)
        sourceerr[t]            += dot(cierr[t],WCI)
    DWIP                        = nutils.sumprod(gierr[1:n],state[:n-1],out=DWIP)
    DWFP                        = nutils.sumprod(gferr[1:n],state[:n-1],out=DWFP)
    DWOP                        = nutils.sumprod(goerr[:n],state[:n],out=DWOP)
    DWGI                        = nutils.sumouter(gierr[:n],source[:n],out=DWGI)
    DWGF                        = nutils.sumouter(gferr[1:n],source[1:n],out=DWGF)
    DWGO                        = nutils.sumouter(goerr[:n],source[:n],out=DWGO)
    DWCI                        = nutils.sumouter(cierr[:n],source[:n],out=DWCI)


def log_mul(x,y):
    return x+y

def log_add(x,y):
    return where(abs(x-y)>10,maximum(x,y),log(exp(clip(x-y,-20,20))+1)+y)

def normalize_nfkc(s):
    return unicodedata.normalize('NFKC',s)

def add_training_info(network):
    return network

def make_target(cs,nc):
    result                      = zeros((2*len(cs)+1,nc))
    for i,j in enumerate(cs):
        result[2*i,0]           = 1.0
        result[2*i+1,j]         = 1.0
    result[-1,0]                = 1.0
    return result

def translate_back0(outputs,threshold=0.25):
    ms                              = amax(outputs,axis=1)
    cs                              = argmax(outputs,axis=1)
    cs[ms<threshold*amax(outputs)]  = 0
    result                          = []
    for i in range(1,len(cs)):
        if cs[i]!=cs[i-1]:
            if cs[i]!=0:
                result.append(cs[i])
    return result

def translate_back(outputs,threshold=0.7,pos=0):
    labels,n                        = measurements.label(outputs[:,0]<threshold)
    mask                            = tile(labels.reshape(-1,1),(1,outputs.shape[1]))
    maxima                          = measurements.maximum_position(outputs,mask,arange(1,amax(mask)+1))
    if pos==1:
        return maxima
    if pos==2: 
        return [(c, outputs[r,c]) for (r,c) in maxima] 
    return [c for (r,c) in maxima] 

def forward_algorithm(match,skip=-5.0):
    v                               = skip*arange(len(match[0]))
    result                          = []
    for i in range(0,len(match)):
        w                           = roll(v,1).copy()
        w[0]                        = skip*i
        v                           = log_add(log_mul(v,match[i]),log_mul(w,match[i]))
        result.append(v)
    return array(result,'f')

def forwardbackward(lmatch):
    lr                              = forward_algorithm(lmatch)
    rl                              = forward_algorithm(lmatch[::-1,::-1])[::-1,::-1]
    both                            = lr+rl
    return both


def ctc_align_targets(outputs,targets,threshold=100.0,verbose=0,debug=0,lo=1e-5):
    outputs                         = maximum(lo,outputs)
    outputs                         = outputs * 1.0/sum(outputs,axis=1)[:,newaxis]
    match                           = dot(outputs,targets.T)
    lmatch                          = log(match)
    assert not isnan(lmatch).any()
    both                            = forwardbackward(lmatch)
    epath                           = exp(both-amax(both))
    l                               = sum(epath,axis=0)[newaxis,:]
    epath                           /= where(l==0.0,1e-9,l)
    aligned                         = maximum(lo,dot(epath,targets))
    l                               = sum(aligned,axis=1)[:,newaxis]
    aligned                         /= where(l==0.0,1e-9,l)
    return aligned

def levenshtein(a,b):
    n, m                            = len(a), len(b)
    if n > m: 
        a,b                         = b,a
        n,m                         = m,n       
    current                         = range(n+1)
    for i in range(1,m+1):
        previous,current            = current,[i]+[0]*n
        for j in range(1,n+1):
            add,delete              = previous[j]+1,current[j-1]+1
            change                  = previous[j-1]
            if a[j-1]!=b[i-1]: 
                change              = change+1
            current[j]              = min(add, delete, change)
    return current[n]

