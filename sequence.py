from lstm import BIDILSTM
from lstm_functions import  normalize_nfkc,\
                            translate_back,\
                            make_target,\
                            ctc_align_targets,\
                            levenshtein
from pylab import array
from scipy.ndimage import filters


class SeqRecognizer:

    def __init__(self,ninput,nstates,noutput=-1,codec=None,normalize=normalize_nfkc):
        self.Ni                             = ninput
        if codec: 
            noutput                         = codec.size()
        assert noutput>0
        self.No                             = noutput
        self.lstm                           = BIDILSTM(ninput,nstates,noutput)
        self.setLearningRate(1e-4)
        self.debug_align                    = 0
        self.normalize                      = normalize
        self.codec                          = codec
        self.clear_log()

    def walk(self):
        for x in self.lstm.walk(): 
            yield x
    
    def clear_log(self):
        self.command_log                    = []
        self.error_log                      = []
        self.cerror_log                     = []
        self.key_log                        = []

    def __setstate__(self,state):
        self.__dict__.update(state)
        self.upgrade()

    def upgrade(self):
        if "last_trial" not in dir(self): 
            self.last_trial                 = 0
        if "command_log" not in dir(self): 
            self.command_log                = []
        if "error_log" not in dir(self): 
            self.error_log                  = []
        if "cerror_log" not in dir(self): 
            self.cerror_log                 = []
        if "key_log" not in dir(self): 
            self.key_log                    = []

    def info(self):
        self.net.info()

    def setLearningRate(self,r,momentum=0.9):
        self.lstm.setLearningRate(r,momentum)

    def predictSequence(self,xs):
        assert xs.shape[1]==self.Ni,\
                            "wrong image height (image: %d, expected: %d)"%(xs.shape[1],self.Ni)
        self.outputs                        = array(self.lstm.forward(xs))
        return translate_back(self.outputs)

    def trainSequence(self,xs,cs,update=1,key=None):
        assert xs.shape[1]==self.Ni,"wrong image height"
        self.outputs                        = array(self.lstm.forward(xs))
        self.targets                        = array(make_target(cs,self.No))
        self.aligned                        = array(ctc_align_targets(self.outputs,self.targets,debug=self.debug_align))
        deltas                              = self.aligned-self.outputs
        self.lstm.backward(deltas)
        if update: 
            self.lstm.update()
        result                              = translate_back(self.outputs)
        self.error                          = sum(deltas**2)
        self.error_log.append(self.error**.5/len(cs))
        self.cerror                         = levenshtein(cs,result)
        self.cerror_log.append((self.cerror,len(cs)))
        self.key_log.append(key)
        return result

    def errors(self,range=10000,smooth=0):
        result                              = self.error_log[-range:]
        if smooth>0: 
            result                          = filters.gaussian_filter(result,smooth,mode='mirror')
        return result

    def cerrors(self,range=10000,smooth=0):
        result                              = [e*1.0/max(1,n) for e,n in self.cerror_log[-range:]]
        if smooth>0: 
            result                          = filters.gaussian_filter(result,smooth,mode='mirror')
        return result

    def s2l(self,s):
        s                                   = self.normalize(s)
        s                                   = [c for c in s]
        return self.codec.encode(s)

    def l2s(self,l):
        l                                   = self.codec.decode(l)
        return u"".join(l)

    def trainString(self,xs,s,update=1):
        return self.trainSequence(xs,self.s2l(s),update=update)

    def predictString(self,xs):
        cs                                  = self.predictSequence(xs)
        return self.l2s(cs)