from setuptools import setup, find_packages

setup(
    name="recon",
    author="Jiabao Lei",
    description="an universal pytorch deep learning experiment codebase",
    url="https://github.com/Karbo123/recon",
    packages=find_packages(),
    install_requires=["gorilla-core", "numpy", "PyYAML"],
    python_requires=">=3.6",
)
