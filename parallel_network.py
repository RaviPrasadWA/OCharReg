from neural_network_base import Network
from pylab import   concatenate,\
                    array 

class Parallel(Network):
    def __init__(self,*nets):
        self.nets       = nets

    def walk(self):
        yield self
        for sub in self.nets:
            for x in sub.walk(): 
                yield x

    def forward(self,xs):
        outputs         = [net.forward(xs) for net in self.nets]
        outputs         = zip(*outputs)
        outputs         = [concatenate(l) for l in outputs]
        return outputs

    def backward(self,deltas):
        deltas          = array(deltas)
        start           = 0
        for i,net in enumerate(self.nets):
            k           = net.noutputs()
            net.backward(deltas[:,start:start+k])
            start       += k
        return None

    def info(self):
        for net in self.nets:
            net.info()
    
    def states(self):
        outputs         = zip(*outputs)
        outputs         = [concatenate(l) for l in outputs]
        return outputs

    def weights(self):
        for i,net in enumerate(self.nets):
            for w,dw,n in net.weights():
                yield w,dw,"Parallel%d/%s"%(i,n)