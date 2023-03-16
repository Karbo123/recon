import setuptools

setuptools.setup(
    name="recon",
    author="Jiabao Lei",
    description="an universal pytorch deep learning experiment codebase",
    url="https://github.com/Karbo123/recon",
    packages=["recon"],
    install_requires=["gorilla-core", "numpy", "PyYAML"],
    python_requires=">=3.6",
)
