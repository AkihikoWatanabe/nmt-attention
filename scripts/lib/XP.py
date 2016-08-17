from chainer import Variable, cuda
import numpy as np
from lib.functions import normalize
from chainer.functions.array.reshape import reshape

class XP:
    __float = None
    __int = None

    @staticmethod
    def set_library(args):
        if args.use_gpu:
            XP.__lib = cuda.cupy
        else:
            XP.__lib = np
        XP.__float = XP.__lib.float32
        XP.__int = XP.__lib.int32

    @staticmethod
    def zeros(shape):
        return Variable(XP.__lib.zeros(shape, dtype=XP.__float))

    @staticmethod
    def ftensor(array, batch_size, height, width):
        normalized = normalize(XP.__lib.array(array, dtype=XP.__float), XP.__lib)
        return Variable(normalized.reshape(batch_size, 1, height, width))

    @staticmethod
    def emtensor(em, batch_size, height, width):
        return reshape(em, (batch_size, 1, height, width))

    @staticmethod
    def iarray(array):
        return Variable(XP.__lib.array(array, dtype=XP.__int)) 
