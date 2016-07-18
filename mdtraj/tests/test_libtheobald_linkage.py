import os.path
import tempfile
#from distutils.core import Extension
#from distutils.dist import Distribution
from setuptools import Distribution, Extension

import mdtraj as md


def test_linkage_libtheobald():
    work_dir = tempfile.mkdtemp()
    src = os.path.join(work_dir, "test.c")
    try:
        with open(src, 'w') as f:
            f.write(""" #include <theobald_rmsd.h>
            #include <center.h>
            int main(int argc, char** argv) {
                float* x, y;
                float z;
                x=0; y=0; z=0;
                inplace_center_and_trace_atom_major(x, &y, 0, 0);
                /// float msd_atom_major(const int nrealatoms, const int npaddedatoms,
                  //   const float* a, const float* b, const float G_a, const float G_b,
                //     int computeRot, float rot[9]);
                msd_atom_major(0 ,0, x, &y, z, z, 0, x);
                return 0;
            } """)
        capi = md.capi()
        ext = Extension(work_dir + '.test_theobald_linkage', [src],
                        include_dirs=[capi['include_dir']],
                        library_dirs=[capi['lib_dir']])
        dist_args = {'ext_modules': [ext]}
        dist = Distribution(dist_args)
        dist.run_command('build_ext')
    finally:
        import shutil
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == '__main__':
    test_linkage_libtheobald()
