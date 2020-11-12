import setuptools


setuptools.setup (
    name = "PyImbalReg", # Replace with your own username
    version = "0.0.1",
    author = "Vahid Asghari",
    author_email = "vd1371@gmail.com",
    description = "Pre-processing technics for imbalanced datasets in regression modelling",
    long_description = open('README.md').read(),
    long_description_content_type = "text/markdown",
    url = "https://github.com/vd1371/PyImbalReg",
    packages = setuptools.find_packages(),
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',

    keywords = [
        
        'PyImbalReg',
        'imblanaced regression',
        'imbalanced data',
        'pre-processing',
        'data augmentation',
        'undersampling',
        'over-sampling',
        'synthetic data',
        'regression'
    ]
)