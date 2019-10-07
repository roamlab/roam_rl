from setuptools import setup, find_packages

setup(
    name="roam_rl",
    version="0.1.0",
    author="Gagan Khandate",
    description="ROAM Lab package for Reinforcement Learning algorithms",
    url="https://github.com/roamlab/roamrl/",
    packages=[package for package in find_packages()
                    if package.startswith('roam_rl')],
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',
        'cython',
        'tqdm',
        'coverage',
        'configparser',
        'tensorflow==1.13.2',
        'gym',
        'baselines',
        ],
    )