import setuptools

# to build and load:
# 1. python setup.py bdist_wheel
# 2. python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/pylineaGT-X.X.X-py3-none-any.whl
# 3. python -m twine upload dist/pylineaGT-X.X.X-py3-none-any.whl

setuptools.setup(
    name = "pybasilica",
    version = "0.0.1",
    author = "Azad Sadr",
    description = "",
    package_dir = {"":"src"},
    packages = setuptools.find_packages(where="src"),
    python_requires=">=3.9",
    install_requires = [
        "pandas==1.4.2",
        "pyro-api==0.1.2",
        "pyro-ppl==1.8.0",
        "pytorch==1.10.2",
        "numpy==1.21.5",
    ],
)