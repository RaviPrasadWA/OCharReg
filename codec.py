import characters

ascii_labels                        = [""," ","~"] + [unichr(x) for x in range(33,126)]

class Codec:

    def init(self,charset):
        charset                     = sorted(list(set(charset)))
        self.code2char              = {}
        self.char2code              = {}
        for code,char in enumerate(charset):
            self.code2char[code]    = char
            self.char2code[char]    = code
        return self

    def size(self):
        return len(list(self.code2char.keys()))

    def encode(self,s):
        dflt                        = self.char2code["~"]
        return [self.char2code.get(c,dflt) for c in s]

    def decode(self,l):
        s                           = [self.code2char.get(c,"~") for c in l]
        return s
        
def ascii_codec():
    return Codec().init(ascii_labels)

def ocr_codec():
    base                            = [c for c in ascii_labels]
    base_set                        = set(base)
    extra                           = [c for c in characters.default if c not in base_set]
    return Codec().init(base+extra)

def getstates_for_display(net):
    if isinstance(net,LSTM):
        return net.state[:net.last_n]
    if isinstance(net,Stacked) and isinstance(net.nets[0],LSTM):
        return net.nets[0].state[:net.nets[0].last_n]
    return None