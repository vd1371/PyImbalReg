import setuptools

setuptools.setup(
    name="PyImbalReg",
    version="0.0.3",
    author="Vahid Asghari",
    author_email="vd1371@gmail.com",
    description="Pre-processing techniques for imbalanced datasets in regression",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/vd1371/PyImbalReg",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    keywords=[
        "PyImbalReg",
        "imbalanced regression",
        "imbalanced data",
        "pre-processing",
        "data augmentation",
        "undersampling",
        "oversampling",
        "synthetic data",
        "regression",
    ],
)