import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="windowagg",
    version="1.0.0",
    author="David Schwartz",
    author_email="dmschwartz@bis.midco.net",
    description="Window based anaylses for geographic information systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/amdenton/SlidingWindows",
    packages=setuptools.find_packages(),
    install_requires=['numpy'],
    license="GNU General Public License v3.0",
)