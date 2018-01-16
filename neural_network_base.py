from pylab import   array,\
                    concatenate,\
                    dot

class Network:

    def predict(self,xs):
        return self.forward(xs)

    def train(self,xs,ys,debug=0):
        xs              = array(xs)
        ys              = array(ys)
        pred            = array(self.forward(xs))
        deltas          = ys - pred
        self.backward(deltas)
        self.update()
        return pred

    def walk(self):
        yield self

    def preSave(self):
        pass

    def postLoad(self):
        pass

    def ctrain(self,xs,cs,debug=0,lo=1e-5,accelerated=1):
        assert len(cs.shape)==1
        assert (cs==array(cs,'i')).all()
        xs                          = array(xs)
        pred                        = array(self.forward(xs))
        deltas                      = zeros(pred.shape)
        assert len(deltas)==len(cs)
        if accelerated:
            if deltas.shape[1]==1:
                for i,c in enumerate(cs):
                    if c==0:
                        deltas[i,0] = -1.0/max(lo,1.0-pred[i,0])
                    else:
                        deltas[i,0] = 1.0/max(lo,pred[i,0])
            else:
                deltas[:,:]         = -pred[:,:]
                for i,c in enumerate(cs):
                    deltas[i,c]     = 1.0/max(lo,pred[i,c])
        else:
            if deltas.shape[1]==1:
                for i,c in enumerate(cs):
                    if c==0:
                        deltas[i,0] = -pred[i,0]
                    else:
                        deltas[i,0] = 1.0-pred[i,0]
            else:
                deltas[:,:]         = -pred[:,:]
                for i,c in enumerate(cs):
                    deltas[i,c]     = 1.0-pred[i,c]
        self.backward(deltas)
        self.update()
        return pred

    def setLearningRate(self,r,momentum=0.9):
        self.learning_rate          = r
        self.momentum               = momentum

    def weights(self):
        pass

    def allweights(self):
        aw                          = list(self.weights())
        weights,derivs,names        = zip(*aw)
        weights                     = [w.ravel() for w in weights]
        derivs                      = [d.ravel() for d in derivs]
        return concatenate(weights),concatenate(derivs)

    def update(self):
        if not hasattr(self,"deltas") or self.deltas is None:
            self.deltas             = [zeros(dw.shape) for w,dw,n in self.weights()]
        for ds,(w,dw,n) in zip(self.deltas,self.weights()):
            ds.ravel()[:]           = self.momentum * ds.ravel()[:] + self.learning_rate * dw.ravel()[:]
            w.ravel()[:]            += ds.ravel()[:]