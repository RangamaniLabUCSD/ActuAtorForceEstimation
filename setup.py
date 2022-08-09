"""PyMem3DG: Automembrane"""

import sys
from setuptools import setup, find_packages
import versioneer

DOCLINES = __doc__.split("\n")

CLASSIFIERS = """\
Development Status :: 3 - Alpha
Environment :: Console
Intended Audience :: Science/Research
License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)
Natural Language :: English
Operating System :: OS Independent
Programming Language :: Python :: 3 :: Only
Topic :: Scientific/Engineering :: Chemistry
Topic :: Scientific/Engineering :: Mathematics
Topic :: Scientific/Engineering :: Physics
Topic :: Scientific/Engineering :: Visualization
"""

# from https://github.com/pytest-dev/pytest-runner#conditional-requirement
needs_pytest = {"pytest", "test", "ptr"}.intersection(sys.argv)
pytest_runner = ["pytest-runner"] if needs_pytest else []

try:
    with open("README.md", "r") as handle:
        long_description = handle.read()
except:
    long_description = None

setup(
    name="automembrane",
    author="The Mem3DG Team",
    author_email="cuzhu@ucsd.edu, ctlee@ucsd.edu",
    description=DOCLINES[0],
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license="MPL2",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.9",
    setup_requires=[] + pytest_runner,
    # url='http://www.my_package.com',  # Website
    # install_requires=[],              # Required packages, pulls from pip if needed; do not use for Conda deployment
    platforms=["Windows", "Linux", "Mac OS-X", "Unix"],
    classifiers=[c for c in CLASSIFIERS.split("\n") if c],
    keywords="meshing membrane mechanics",
    zip_safe=True,
)
