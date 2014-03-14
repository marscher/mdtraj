##############################################################################
# MDTraj: A Python Library for Loading, Saving, and Manipulating
#         Molecular Dynamics Trajectories.
# Copyright 2012-2013 Stanford University and the Authors
#
# Authors: Robert McGibbon
# Contributors:
#
# MDTraj is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 2.1
# of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with MDTraj. If not, see <http://www.gnu.org/licenses/>.
##############################################################################


##############################################################################
# Imports
##############################################################################
import cython
import numpy as np
from mdtraj.utils import ensure_type
import scipy.spatial.distance

cimport numpy as np
from cpython cimport bool
from cython.parallel cimport prange

np.import_array()

cdef extern from "include/Munkres.h":
    cdef cppclass Munkres:
        Munkres()
        void solve(double* icost, int* answer, int m, int n)

def lprmsd(target, reference, int frame=0, atom_indices=None, permute_indices=None,
           bool parallel=True):
    """rmsd(target, reference, frame=0, atom_indices=None, parallel=True, precentered=False)

    Compute LP-RMSD of all conformations in target to a reference conformation.
    The LP-RMSD is the minimum root-mean squared deviation between two sets of
    points, minimizing over both the rotational/translational degrees of freedom
    AND the label correspondences between points in the target and reference
    conformations. This means that it can be used meaningfully with atoms with
    exchange symmetry like, like multiple water molecules.

    Parameters
    ----------
    target : md.Trajectory
        For each conformation in this trajectory, compute the RMSD to
        a particular 'reference' conformation in another trajectory
        object.
    reference : md.Trajectory
        The object containing the reference conformation to measure distances
        to.
    frame : int
        The index of the conformation in `reference` to measure
        distances to.
    atom_indices : array_like, or None
        The indices of the atoms to use in the RMSD calculation. If not
        supplied, all atoms will be used.
    permute_indices : list of array_like, or None
        A list of groups of permutable atoms. Each element in permute_indices
        is an array of indices containing atoms whose labels can be mutually
        exchanged within the group. If none, all points in atom_indices
        will be allowed each other.
    parallel : bool
        Use OpenMP to calculate each of the RMSDs in parallel over
        multiple cores.
    """
    if atom_indices is None:
        atom_indices = np.arange(reference.n_atoms, dtype=np.int)
    else:
        atom_indices = ensure_type(np.asarray(atom_indices), dtype=np.int, ndim=1, name='atom_indices')
        if not np.all((atom_indices >= 0) * (atom_indices < target.xyz.shape[1]) * (atom_indices < reference.xyz.shape[1])):
            raise ValueError("atom_indices must be valid positive indices")
    if permute_indices is None:
        permute_indices = [atom_indices]
    else:
        permute_indices = [ensure_type(np.asarray(group), dtype=np.int, ndim=1, name='permute_indices[%d]' % i) for i, group in enumerate(permute_indices)]

    assert (target.xyz.ndim == 3) and (reference.xyz.ndim == 3) and (target.xyz.shape[2]) == 3 and (reference.xyz.shape[2] == 3)
    if not (target.xyz.shape[1]  == reference.xyz.shape[1]):
        raise ValueError("Input trajectories must have same number of atoms. "
                         "found %d and %d." % (target.xyz.shape[1], reference.xyz.shape[1]))
    if frame >= reference.xyz.shape[0]:
        raise ValueError("Cannot calculate RMSD of frame %d: reference has "
                         "only %d frames." % (frame, reference.xyz.shape[0]))



def compute_permutation(np.ndarray[ndim=2, dtype=np.float32_t, mode='c'] target,
                        np.ndarray[ndim=2, dtype=np.float32_t, mode='c'] reference,
                        permute_indices=None):
    """compute_permutation(target, reference, permute_indices)

    Solve the assignment problem, finding the bijection of indices that minimizes
    the sum of the euclidean distance between the all points in `target`
    and `reference`.

    This routine does not rotate either the coordinates in `target` or
    `reference`.

    Parameters
    ----------
    target : np.ndarray, ndim=2
        The cartesian coordinates of a set of atoms -- a single conformation
    reference : np.ndarray, ndim=2
        The cartesian coordinates of a set of atoms -- a single conformation
    permute_indices : list of array_like, or None
        A list of groups of permutable atoms. Each element in permute_indices
        is an array of indices containing atoms whose labels can be mutually
        exchanged within the group. If none, all points will be allowed to
        map to all other points (a single group containing all the points).

    Returns
    -------
    mapping :
    """
    if target.shape[0] != reference.shape[0] or target.shape[1] != reference.shape[1]:
        raise ValueError('target (shape=(%d,%d)) and reference (shape=%d, %d) must have the saame dimensions' % (
            target.shape[0], target.shape[1], reference.shape[0], reference.shape[1]))

    if permute_indices is None:
        permute_indices = [np.arange(len(target))]
    else:
        permute_indices = [ensure_type(np.asarray(group), dtype=np.int, ndim=1, name='permute_indices[%d]' % i) for i, group in enumerate(permute_indices)]

    cdef int i, j
    cdef int n_atoms = target.shape[0]
    cdef np.ndarray[ndim=2, dtype=np.double_t, mode='c'] distance
    cdef np.ndarray[ndim=2, dtype=np.int32_t, mode='c'] mask


    # use only the distances in permute_indices actually... (set others to inf)
    distance = scipy.spatial.distance.cdist(target, reference)

    mask = _munkres(distance)
    mapping = np.empty(n_atoms, dtype=np.int32)

    for i in range(n_atoms):
        for j in range(n_atoms):
            if mask[j, i]:
                mapping[i] = j
                break


    return mapping


@cython.boundscheck(False)
def _munkres(np.ndarray[np.double_t, ndim=2, mode="c"] A not None):
    """_munkres(A)

    Calculate the minimum cost assignment of a cost matrix, A

    Parameters
    ----------
    A : np.ndarray, dtype=np.double, ndim=2

    Returns
    -------
    assignments : np.ndarray, ndim=2, dtype=int32
        Boolean array with shape equal to the shape of A. assignments[i,j] == 1
        for an assignment, and 0 for a non-assignment

    Examples
    -------
    >>> _munkres(np.array([[7, 4, 3], [6, 8, 5], [9, 4, 4]], dtype=np.double))
    [[0 0 1],
     [1 0 0],
     [0 1 0]]
    """
    cdef int x = A.shape[0]
    cdef int y = A.shape[1]
    cdef np.ndarray[ndim=2, dtype=np.int32_t, mode='c'] rslt

    rslt = np.zeros(shape=(x,y), dtype=np.int32, order='c')
    cdef Munkres* munk = new Munkres()
    munk.solve(<double *> A.data, <int *> rslt.data, x, y)
    del munk

    return rslt
