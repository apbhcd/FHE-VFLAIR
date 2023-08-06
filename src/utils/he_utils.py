import math
import numpy as np
from Pyfhel import Pyfhel

schemeDict = {
        "1024": {'scheme': 'CKKS', 'n': 2**10,  'scale': 2**20,  'qi_sizes': [27]},
        "2048": {'scheme': 'CKKS', 'n': 2**11,  'scale': 2**20,  'qi_sizes': [54]},
        "4096": {'scheme': 'CKKS', 'n': 2**12,  'scale': 2**20,  'qi_sizes': [30]},
        "8192": {'scheme': 'CKKS', 'n': 2**13, 'scale': 2**20,  'qi_sizes': [30]},
        "16384": {'scheme': 'CKKS', 'n': 2**14, 'scale': 2**40,  'qi_sizes': [60]},
        "32768": {'scheme': 'CKKS', 'n': 2**15, 'scale': 2**30,  'qi_sizes': [60]}
}

def generate_ckks_key(np):
    HE = Pyfhel()
    ckks_params = schemeDict.get(np)
    status = HE.contextGen(**ckks_params)
    print(status)
    HE.keyGen()
    return HE


def enc_vector(HE, arr_x):

    ptxt_x = HE.encodeFrac(arr_x)
    ctxt_x = HE.encryptPtxt(ptxt_x) 
    return ctxt_x


def dec_vector(HE, ctxt_x):
    r_x = HE.decryptFrac(ctxt_x)
    _r = lambda x: np.round(x, decimals=3)
    return _r(r_x)
