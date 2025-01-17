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
        "earthengine-api==0.1.390",
        "geemap==0.31.0",
        "geopandas==0.14.2",
        "numpy==1.26.4",
        "pandas==2.0.0"
    ]
)