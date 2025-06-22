import pyttb as ttb
import numpy as np
import copy as cpy
import goated.rol_interface.TuckerVector as tv
import goated.rol_interface.CPVector as cv


import logging
class ExtraCopyFilter(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith("Selected no copy, but input data isn't")
logger = logging.getLogger()  # root logger
logger.addFilter(ExtraCopyFilter())


def vec_to_array(x):
    y = np.reshape(x.core,(-1,),order='F')
    for f in x.factors:
        y = np.concat([y,np.reshape(f,(-1,),order='F')])
    return y


def rolvec_to_ttensor(x, copy=False):
    return ttb.ttensor(ttb.tensor(x.core), x.factors, copy=copy)


def rolvec_to_ktensor(x, copy=False):
    # if copy:
    #     return ttb.ktensor(cpy.deepcopy(x.data))
    # else:
    #     return ttb.ktensor(x.data)
    return ttb.ktensor(x.data, copy=copy)


def ttensor_to_rolvec(x, copy=False):
    if copy:
        return tv.TuckerVector(cpy.deepcopy(x.core.data), cpy.deepcopy(x.factor_matrices))
    else:
        return tv.TuckerVector(x.core.data, x.factor_matrices)


def ktensor_to_rolvec(x, copy=False):
    if copy:
        return cv.CPVector(cpy.deepcopy(x.factor_matrices))
    else:
        return cv.CPVector(x.factor_matrices)
