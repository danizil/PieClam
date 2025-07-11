from setuptools import setup, find_packages

setup(
    name="PieClam",
    version="0.1.0",
    description="Graphon learning and link prediction experiments (ICML PieClam project)",
    author="Danny Zilberg",
    author_email="dannyzilberg@gmail.com",
    packages=find_packages(),
    install_requires=[],
    include_package_data=True,
    python_requires=">=3.6",
)