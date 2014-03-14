from __future__ import print_function
from mdtraj.testing import eq
import numpy as np
from mdtraj._lprmsd import _munkres, compute_permutation

def test_munkres_0():
    result = _munkres(np.array([[7, 4, 3], [6, 8, 5], [9, 4, 4]], dtype=np.double))
    true = np.array([[0, 0, 1],
                     [1, 0, 0],
                     [0, 1, 0]], dtype=np.int32)
    eq(result, true)

def test_compute_permutation_0():
    reference = np.array([0, 1, 2, 3, 4, 5], dtype=np.float32).reshape(6, 1)
    target = np.array([1, 2, 3, 4, 5, 0], dtype=np.float32).reshape(6, 1)

    permutation = compute_permutation(target, reference)
    cost = np.sum(reference[permutation] - target)
    eq(cost, np.float32(0.0))
