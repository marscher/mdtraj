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

cimport numpy as np
from cpython cimport bool
from cython.parallel cimport prange

np.import_array()

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



def compute_permutation(np.ndarray[ndim=2, dtype=np.float32_t, mode='c'] target_xyz,
                        np.ndarray[ndim=2, dtype=np.float32_t, mode='c'] reference_xyz,
                        list permute_indices=None):
    """compute_permutation(target_xyz, reference_xyz, permute_indices)

    Solve the assignment problem, finding the bijection of indices that minimizes
    the sum of the euclidean distance between the all points in `target_xyz`
    and `reference_xyz`.

    This routine does not rotate either the coordinates in `target_xyz` or
    `reference_xyz`.

    Parameters
    ----------
    target_xyz : np.ndarray, ndim=2
        The cartesian coordinates of a set of atoms -- a single conformation
    reference_xyz : np.ndarray, ndim=2
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
    if permute_indices is None:
        raise ValueError('permute_indices is a required argument')
    else:
        permute_indices = [ensure_type(np.asarray(group), dtype=np.int, ndim=1, name='permute_indices[%d]' % i) for i, group in enumerate(permute_indices)]

    
