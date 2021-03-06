"""MDTraj: a python library for loading, saving, and manipulating molecular dynamics trajectories.

MDTraj provides an easy to use python interface for manipulating MD trajectories.
It supports the reading and writing of molecular dynamics trajectories
in a variety of formats, including full support for PDB, DCD, XTC, TRR, binpos,
AMBER NetCDF, AMBER mdcrd and MDTraj HDF5. The package also provides a command line script
for converting trajectories between supported formats.
"""

from __future__ import print_function
DOCLINES = __doc__.split("\n")

import os
import sys
import shutil
import tempfile
import subprocess
from distutils.ccompiler import new_compiler
try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension

try:
    import numpy
except ImportError:
    print('Building and running mdtraj requires numpy', file=sys.stderr)
    sys.exit(1)

try:
    import Cython
    if Cython.__version__ < '0.18':
        raise ImportError
    from Cython.Distutils import build_ext
    setup_kwargs = {'cmdclass': {'build_ext': build_ext}}
except ImportError:
    print('Building from source requires cython >= 0.18', file=sys.stderr)
    exit(1)

try:
    sys.argv.remove('--no-install-deps')
    no_install_deps = True
except ValueError:
    no_install_deps = False

if 'setuptools' in sys.modules:
    setup_kwargs['zip_safe'] = False
    if not no_install_deps:
        setup_kwargs['install_requires'] = ['pandas>=0.9.0']
    setup_kwargs['entry_points'] = {'console_scripts':
              ['mdconvert = mdtraj.scripts.mdconvert:entry_point',
               'mdinspect = mdtraj.scripts.mdinspect:entry_point']}
else:
    setup_kwargs['scripts'] = ['scripts/mdconvert.py', 'scripts/mdinspect.py']


##########################
VERSION = "0.8.1"
ISRELEASED = False
__version__ = VERSION
##########################


CLASSIFIERS = """\
Development Status :: 3 - Alpha
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: GNU Lesser General Public License v2 or later (LGPLv2+)
Programming Language :: C
Programming Language :: Python
Programming Language :: Python :: 3
Topic :: Scientific/Engineering :: Bio-Informatics
Topic :: Scientific/Engineering :: Chemistry
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""

def find_packages():
    """Find all of mdtraj's python packages.
    Adapted from IPython's setupbase.py. Copyright IPython
    contributors, licensed under the BSD license.
    """
    packages = ['mdtraj.scripts']
    for dir,subdirs,files in os.walk('MDTraj'):
        package = dir.replace(os.path.sep, '.')
        if '__init__.py' not in files:
            # not a package
            continue
        packages.append(package.replace('MDTraj', 'mdtraj'))
    return packages


################################################################################
# Writing version control information to the module
################################################################################

def git_version():
    # Return the git revision as a string
    # copied from numpy setup.py
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = 'Unknown'

    return GIT_REVISION


def write_version_py(filename='MDTraj/version.py'):
    cnt = """
# THIS FILE IS GENERATED FROM MDTRAJ SETUP.PY
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
git_revision = '%(git_revision)s'
release = %(isrelease)s

if not release:
    version = full_version
"""
    # Adding the git rev number needs to be done inside write_version_py(),
    # otherwise the import of numpy.version messes up the build under Python 3.
    FULLVERSION = VERSION
    if os.path.exists('.git'):
        GIT_REVISION = git_version()
    else:
        GIT_REVISION = 'Unknown'

    if not ISRELEASED:
        FULLVERSION += '.dev-' + GIT_REVISION[:7]

    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION,
                       'full_version': FULLVERSION,
                       'git_revision': GIT_REVISION,
                       'isrelease': str(ISRELEASED)})
    finally:
        a.close()

################################################################################
# Detection of compiler capabilities
################################################################################

def hasfunction(cc, funcname, include=None, extra_postargs=None):
    # From http://stackoverflow.com/questions/
    #            7018879/disabling-output-when-compiling-with-distutils
    tmpdir = tempfile.mkdtemp(prefix='hasfunction-')
    devnull = oldstderr = None
    try:
        try:
            fname = os.path.join(tmpdir, 'funcname.c')
            f = open(fname, 'w')
            if include is not None:
                f.write('#include %s\n' % include)
            f.write('int main(void) {\n')
            f.write('    %s;\n' % funcname)
            f.write('}\n')
            f.close()
            devnull = open(os.devnull, 'w')
            oldstderr = os.dup(sys.stderr.fileno())
            os.dup2(devnull.fileno(), sys.stderr.fileno())
            objects = cc.compile([fname], output_dir=tmpdir,
                                 extra_postargs=extra_postargs)
            cc.link_executable(objects, os.path.join(tmpdir, 'a.out'))
        except Exception as e:
            return False
        return True
    finally:
        if oldstderr is not None:
            os.dup2(oldstderr, sys.stderr.fileno())
        if devnull is not None:
            devnull.close()
        shutil.rmtree(tmpdir)


def detect_openmp():
    "Does this compiler support OpenMP parallelization?"
    compiler = new_compiler()
    print('Attempting to autodetect OpenMP support...', end=' ')
    hasopenmp = hasfunction(compiler, 'omp_get_num_threads()')
    needs_gomp = hasopenmp
    if not hasopenmp:
        compiler.add_library('gomp')
        hasopenmp = hasfunction(compiler, 'omp_get_num_threads()')
        needs_gomp = hasopenmp
    print
    if hasopenmp:
        print('Compiler supports OpenMP')
    else:
        print('Did not detect OpenMP support; parallel RMSD disabled')
    return hasopenmp, needs_gomp


def detect_sse3():
    "Does this compiler support SSE3 intrinsics?"
    compiler = new_compiler()
    return hasfunction(compiler, '__m128 v; _mm_hadd_ps(v,v)',
                       include='<pmmintrin.h>',
                       extra_postargs=['-msse3'])

def detect_sse41():
    "Does this compiler support SSE4.1 intrinsics?"
    compiler = new_compiler()
    return hasfunction(compiler, '__m128 v; _mm_round_ps(v,0x00)',
                       include='<smmintrin.h>',
                       extra_postargs=['-msse4'])

################################################################################
# Declaration of the compiled extension modules (cython + c)
################################################################################


xtc = Extension('mdtraj.formats.xtc',
                sources=['MDTraj/formats/xtc/src/xdrfile.c',
                         'MDTraj/formats/xtc/src/xdrfile_xtc.c',
                         'MDTraj/formats/xtc/xtc.pyx'],
                include_dirs=['MDTraj/formats/xtc/include/',
                              'MDTraj/formats/xtc/', numpy.get_include()])

trr = Extension('mdtraj.formats.trr',
                sources=['MDTraj/formats/xtc/src/xdrfile.c',
                         'MDTraj/formats/xtc/src/xdrfile_trr.c',
                         'MDTraj/formats/xtc/trr.pyx'],
                include_dirs=['MDTraj/formats/xtc/include/',
                              'MDTraj/formats/xtc/', numpy.get_include()])

dcd = Extension('mdtraj.formats.dcd',
                sources=['MDTraj/formats/dcd/src/dcdplugin.c',
                         'MDTraj/formats/dcd/dcd.pyx'],
                include_dirs=["MDTraj/formats/dcd/include/",
                              'MDTraj/formats/dcd/', numpy.get_include()])

binpos = Extension('mdtraj.formats.binpos',
                   sources=['MDTraj/formats/binpos/src/binposplugin.c',
                            'MDTraj/formats/binpos/binpos.pyx'],
                   include_dirs=['MDTraj/formats/binpos/include/',
                                 'MDTraj/formats/binpos/', numpy.get_include()])


def rmsd_extension():
    openmp_enabled, needs_gomp = detect_openmp()
    compiler_args = ['-msse2' if not detect_sse3() else '-mssse3',
                     '-O3', '-funroll-loops']
    if new_compiler().compiler_type == 'msvc':
        compiler_args.append('/arch:SSE2')

    if openmp_enabled:
        compiler_args.append('-fopenmp')
    compiler_libraries = ['gomp'] if needs_gomp else []
    #compiler_defs = [('USE_OPENMP', None)] if openmp_enabled else []

    rmsd = Extension('mdtraj._rmsd',
                     sources=[
                         'MDTraj/rmsd/src/theobald_rmsd.c',
                         'MDTraj/rmsd/src/rotation.c',
                         'MDTraj/rmsd/src/center.c',
                         'MDTraj/rmsd/_rmsd.pyx'],
                     include_dirs=[
                         'MDTraj/rmsd/include', numpy.get_include()],
                     extra_compile_args=compiler_args,
                     #define_macros=compiler_defs,
                     libraries=compiler_libraries)
    return rmsd


def geometry():
    if not detect_sse3():
        return None

    extra_compile_args = ['-mssse3']
    define_macros = []
    if detect_sse41():
        define_macros.append(('__SSE4__', 1))
        define_macros.append(('__SSE4_1__', 1))
        extra_compile_args.append('-msse4')

    return Extension('mdtraj.geometry._geometry',
                     sources=['MDTraj/geometry/src/geometry.c',
                              'MDTraj/geometry/src/sasa.c',
                              'MDTraj/geometry/src/_geometry.pyx'],
                     include_dirs=['MDTraj/geometry/include', numpy.get_include()],
                     define_macros=define_macros,
                     extra_compile_args=extra_compile_args)

extensions = [xtc, trr, dcd, binpos, rmsd_extension()]
ext = geometry()
if ext is not None:
    extensions.append(ext)

write_version_py()
setup(name='mdtraj',
      author='Robert McGibbon',
      author_email='rmcgibbo@gmail.com',
      description=DOCLINES[0],
      long_description="\n".join(DOCLINES[2:]),
      version=__version__,
      license='LGPLv2.1+',
      url='http://rmcgibbo.github.io/mdtraj',
      platforms=['Linux', 'Mac OS-X', 'Unix', 'Windows'],
      classifiers=CLASSIFIERS.splitlines(),
      packages=find_packages(),
      package_dir={'mdtraj': 'MDTraj', 'mdtraj.scripts': 'scripts'},
      ext_modules=extensions,
      package_data={'mdtraj.formats.pdb': ['data/*'],
                    'mdtraj.testing': ['reference/*']},
      **setup_kwargs)
