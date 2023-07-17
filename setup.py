from setuptools import setup

setup(
    name='evodm', 
    version='1.0.0', 
    author = 'Davis Weaver', 
    author_email = 'dtw43@case.edu',
    packages=['evodm', "evodm.test"], 
    install_requires = [
      "tensorflow~=2.12.0", 
      "keras~=2.12.0",
      "numpy",
      "tqdm",
      "scipy", 
      "networkx", 
      "pytest", 
      "pymdptoolbox", 
      "pandas", 
      "matplotlib", 
      "ruptures"
    ],
)