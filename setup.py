from setuptools import setup

setup(
    name='evodm', 
    version='1.1.0', 
    author = 'Davis Weaver', 
    author_email = 'dtw43@case.edu',
    packages=['evodm', "evodm.test"], 
    install_requires = [
      "tensorflow~=2.11.0", 
      "numpy",
      "scipy>=1.8.0", 
      "networkx", 
      "pytest", 
      "tqdm",
      "pymdptoolbox", 
      "pandas", 
      "matplotlib", 
      "ruptures"
    ],
)