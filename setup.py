from setuptools import setup, find_packages

setup(
    name="roam_rl",
    version="1.0.0",
    author="Gagan Khandate",
    description="ROAM Lab package for Reinforcement Learning algorithms",
    url="https://github.com/roamlab/roamrl/",
    packages=[package for package in find_packages()
                    if package.startswith('roam_rl')],
    # Note: install requires does not list all the dependencies for full functionality
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',
        'tqdm',
        'configparser',
        ],
    )