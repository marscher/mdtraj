from mdtraj.testing import eq
import numpy as np
from mdtraj._lprmsd import _munkres

def test_munkres_0():
    result = _munkres(np.array([[7, 4, 3], [6, 8, 5], [9, 4, 4]], dtype=np.double))
    true = np.array([[0, 0, 1],
                     [1, 0, 0],
                     [0, 1, 0]], dtype=np.int32)
    eq(result, true)
