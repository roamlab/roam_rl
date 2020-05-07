from setuptools import setup, find_packages

setup(
    name="roam_rl",
    version="0.0.1",
    author="Gagan Khandate",
    description= "Helper classes for openai/baselines",
    url="https://github.com/roamlab/roamrl/",
    packages=[package for package in find_packages()
                    if package.startswith('roam_rl')],
    # Note: install requires does not list all the dependencies for full functionality
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',
        ],
    )
