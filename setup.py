import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="spyplotter",
    version="0.0.1",
    author="Elisa Schoesser",
    author_email="elisa.schoesser@uni-heidelberg.de",
    packages=["spyplotter"],
    description="Plotting tools for quantitative spectroscopy with focus on atmosphere models as PoWR and CMFGEN",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eschoesser/spyplotter",
    license='GPT',
    python_requires='>=3.6',
    install_requires=[
         "numpy",
         "matplotlib",
         "ipympl",
         "scipy",
         "pandas"
    ]
)