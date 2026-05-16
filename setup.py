from setuptools import setup, find_packages

setup(
    name="phino",
    version="0.1.0",
    description="PHINO: Physical Intuition via Newtonian Observations",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "timm>=1.0.3",
        "transformers>=4.40.0",
        "einops>=0.7.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0",
        "scikit-learn>=1.4.0",
        "Pillow>=10.0.0",
    ],
)
