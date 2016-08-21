from chainer import Variable, cuda
import numpy as np

class XP:
    __float = None
    __int = None

    @staticmethod
    def set_library(use_gpu):
        if use_gpu:
            XP.__lib = cuda.cupy
        else:
            XP.__lib = np
        XP.__float = XP.__lib.float32
        XP.__int = XP.__lib.int32

    @staticmethod
    def zeros(shape):
        return Variable(XP.__lib.zeros(shape, dtype=XP.__float))

    @staticmethod
    def iarray(array):
        return Variable(XP.__lib.array(array, dtype=XP.__int)) 
