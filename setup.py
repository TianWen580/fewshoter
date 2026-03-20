#!/usr/bin/env python3
"""
Setup script for CLIP Few-Shot Classification System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = (
        "CLIP Few-Shot Fine-Grained Bird and Animal Classification System"
    )

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#") and not line.startswith("git+")
        ]
else:
    requirements = [
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "opencv-python>=4.5.0",
        "Pillow>=8.0.0",
        "numpy<=1.26.1",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "pandas>=1.3.0",
        "tqdm>=4.62.0",
        "PyYAML>=5.4.0",
        "imageio>=2.9.0",
    ]

setup(
    name="fewshoter",
    version="1.0.0",
    author="CLIP Few-Shot Classification Team",
    description="CLIP-based few-shot image classification toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["fewshoter"] + [f"fewshoter.{pkg}" for pkg in find_packages()],
    package_dir={"fewshoter": "."},
    python_requires=">=3.8",
    install_requires=requirements,
    license="MIT",
    url="https://github.com/TianWen580/fewshoter",
    project_urls={
        "Source": "https://github.com/TianWen580/fewshoter",
        "Issues": "https://github.com/TianWen580/fewshoter/issues",
        "Documentation": "https://github.com/TianWen580/fewshoter#readme",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "fewshoter-train=fewshoter.cli.train:main",
            "fewshoter-inference=fewshoter.cli.inference:main",
            "fewshoter-evaluate=fewshoter.cli.evaluate:main",
            "clip-fewshot-train=fewshoter.cli.train:main",
            "clip-fewshot-inference=fewshoter.cli.inference:main",
            "clip-fewshot-evaluate=fewshoter.cli.evaluate:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
