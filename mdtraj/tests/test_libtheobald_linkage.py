import os.path
import tempfile

from setuptools import Distribution, Extension

import mdtraj as md

pyx_src = """
cdef extern from "center.h":
    void inplace_center_and_trace_atom_major(float* coords, float* traces,
    const int n_frames, const int n_atoms)

cdef extern from "theobald_rmsd.h":
    float msd_atom_major(const int nrealatoms, const int npaddedatoms,
                     const float* a, const float* b, const float G_a, const float G_b,
                     int computeRot, float rot[9]);

def test():
    cdef float* x;
    cdef float* y;
    cdef float z;
    x=NULL; y=NULL; z=0;
    inplace_center_and_trace_atom_major(x, y, 0, 0);
    msd_atom_major(0 ,0, x, y, z, z, 0, x);

"""

def test_linkage_libtheobald():
    try:
        from Cython.Build import cythonize
    except ImportError:
        import warnings
        warnings.warn("test_linkage_libtheobald skipped because Cython is missing.")
        return

    work_dir = tempfile.mkdtemp()
    src = os.path.join(work_dir, "test.pyx")
    try:
        with open(src, 'w') as f:
            f.write(pyx_src)
        capi = md.capi()
        ext = Extension(work_dir + '.test_theobald_linkage', [src],
                        include_dirs=[capi['include_dir']],
                        library_dirs=[capi['lib_dir']])
        dist_args = {'ext_modules': cythonize([ext])}
        dist = Distribution(dist_args)
        dist.run_command('build_ext')
    finally:
        import shutil
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == '__main__':
    test_linkage_libtheobald()
