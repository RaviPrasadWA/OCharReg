from neural_network_base import Network

class Reversed(Network):

    def __init__(self,net):
        self.net = net

    def walk(self):
        yield self
        for x in self.net.walk(): yield x

    def ninputs(self):
        return self.net.ninputs()

    def noutputs(self):
        return self.net.noutputs()

    def forward(self,xs):
        return self.net.forward(xs[::-1])[::-1]

    def backward(self,deltas):
        result = self.net.backward(deltas[::-1])
        return result[::-1] if result is not None else None

    def info(self):
        self.net.info()

    def states(self):
        return self.net.states()[::-1]

    def weights(self):
        for w,dw,n in self.net.weights():
            yield w,dw,"Reversed/%s"%n