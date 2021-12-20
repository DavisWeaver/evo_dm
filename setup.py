from setuptools import setup

setup(
    name='evodm', 
    version='0.1.0', 
    author = 'Davis Weaver', 
    author_email = 'dtw43@case.edu',
    packages=['evodm', "evodm.test"], 
    install_requires = [
      "tensorflow~=2.6.1", 
      "keras~=2.6.0",
      "numpy",
      "tqdm",
      "scipy", 
      "networkx", 
      "matplotlib~=3.3.4", 
      "pytest", 
      "pymdptoolbox"
    ],
    python_requires='>= 3.6.0, <=3.9.11'
)