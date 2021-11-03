from setuptools import setup

setup(
    name='evodm', 
    version='0.1.0', 
    author = 'Davis Weaver', 
    author_email = 'dtw43@case.edu',
    packages=['evodm', "evodm.test"], 
    install_requires = [
      "tensorflow >= 2.50", 
      "keras",
      "numpy",
      "tqdm",
      "scipy", 
      "networkx", 
      "matplotlib", 
      "pytest", 
    ],
)