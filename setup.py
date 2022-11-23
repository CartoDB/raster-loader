from os.path import join
from setuptools import find_packages, setup

version_ns = {}
with open(join("raster_loader", "_version.py")) as f:
    exec(f.read(), {}, version_ns)

setup(
    name="raster-loader",
    version=version_ns["__version__"],
    description="Python library to authenticate with CARTO",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords=["carto", "auth", "oauth", "carto-dw", "bigquery"],
    author="CARTO",
    author_email="jarroyo@carto.com",
    url="https://github.com/cartodb/raster-loader",
    license="BSD 3-Clause",
    packages=find_packages(exclude=["examples", "tests"]),
    python_requires=">=3.7",
    install_requires=["requests", "pyyaml"],
    extras_require={"carto-dw": ["google-auth"]},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    zip_safe=False,
)
