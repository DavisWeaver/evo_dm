from setuptools import setup

setup(
    name='evodm', 
    version='0.1.0', 
    author = 'Davis Weaver', 
    author_email = 'dtw43@case.edu',
    packages=['evodm', "evodm.test"], 
    install_requires = [
      "tensorflow", 
      "keras",
      "numpy",
      "tqdm",
      "scipy", 
      "networkx", 
      "matplotlib", 
      "pytest", 
      "pandas", 
      "math"
    ],
)