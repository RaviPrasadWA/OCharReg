from neural_network_base import Network
from lstm_functions import  randu,\
                            sigmoid,\
                            sumouter,\
                            initial_range
from pylab import   zeros,\
                    concatenate,\
                    dot,\
                    array,\
                    rand

class Logreg(Network):

    def __init__(self,Nh,No,initial_range=initial_range,rand=rand):
        self.Nh                     = Nh
        self.No                     = No
        self.W2                     = randu(No,Nh+1)*initial_range
        self.DW2                    = zeros((No,Nh+1))

    def ninputs(self):
        return self.Nh

    def noutputs(self):
        return self.No

    def forward(self,ys):
        n                           = len(ys)
        inputs,zs                   = [None]*n,[None]*n
        for i in range(n):
            inputs[i]               = concatenate([ones(1),ys[i]])
            zs[i]                   = sigmoid(dot(self.W2,inputs[i]))
        self.state                  = (inputs,zs)
        return zs

    def backward(self,deltas):
        inputs,zs                   = self.state
        n                           = len(zs)
        assert len(deltas)==len(inputs)
        dzspre,dys                  = [None]*n,[None]*n
        for i in reversed(range(len(zs))):
            dzspre[i]               = deltas[i] * zs[i] * (1-zs[i])
            dys[i]                  = dot(dzspre[i],self.W2)[1:]
        self.dzspre                 = dzspre
        self.DW2                    = sumouter(dzspre,inputs)
        return dys

    def info(self):
        vars_                       = sorted("W2".split())
        for v in vars_:
            a                       = array(getattr(self,v))

    def weights(self):
        yield self.W2,self.DW2,"Logreg"