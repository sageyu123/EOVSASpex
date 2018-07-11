from collections import OrderedDict
class Layer:
    # alpha = 1.0
    # dmax = None
    # dmin = None

    # @classmethod
    # def set_alpha(cls, alp):
    #     if 0.0 <= alp <= 1.0:
    #         cls.alpha = alp
    #     else:
    #         print('alpha value must in range of [0,1]')

    def __init__(self, name, url, idx):
        self.alpha = 1.0
        self.layer_name = 'layer{}'.format(name)
        self.url = url
        self.idx = idx
        self.nidx = len(url)
        self.bound = [0, -1]

    @property
    def info(self):
        # od =OrderedDict()
        # od['layer_name']=self.layer_name
        # od['url']= self.url
        # od['idx']= 0
        # od['nidx']= self.nidx
        # od['bound']= self.bound
        # return od
        return vars(self)

    def __repr__(self):
        return "Layer('{}')".format(self.layer_name)
