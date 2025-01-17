from setuptools import setup, find_packages

setup(
    name="my_gee_package",
    version="0.0.1",
    description="A package that extends the functionality of Google Earth Engine's Python API.",
    author="Thomas Lauber",
    author_email="thomas.lauber@usys.ethz.ch",
    url="https://github.com/thomaslauber/geePipe",
    packages=find_packages(),
    install_requires=[
        "earthengine-api>=0.1.0",  # Ensure the user has GEE API installed
        "numpy",  # Add other dependencies as needed
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)