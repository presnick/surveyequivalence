import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="survey-equivalence-presnick", # Replace with your own username
    version="0.0.1",
    author="Paul Resnick, Tim Weninger, Yuqing Kong, and Grant Schoenebeck",
    author_email="presnick@umich.edu",
    description="Code for calculating survey equivalence",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/umsi/surveyequivalence",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)