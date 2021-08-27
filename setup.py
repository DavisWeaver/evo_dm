from setuptools import setup

setup(
    name='evo_dm', 
    version='0.1.0', 
    author = 'Davis Weaver', 
    author_email = 'dtw43@case.edu',
    packages=['evo_dm', "evo_dm.test"], 
    install_requires = [
      "tensorflow", 
      "numpy",
      "tqdm",
      "scipy", 
      "networkx", 
      "matplotlib", 
      "pytest"
    ],
)