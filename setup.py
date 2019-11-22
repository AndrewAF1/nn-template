import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytorch nn templates",
    version="0.0.1",
    author="Andrew Farabow",
    author_email="aafarabow@gmail.com",
    description="Some neural nets for various things",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/andrewaf1/pytorch-nn-templates",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
