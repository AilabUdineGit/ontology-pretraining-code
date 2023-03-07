from pathlib import Path

from setuptools import find_namespace_packages, setup


def requirements():
    BASE_DIR = Path(__file__).parent
    with open(Path(BASE_DIR, "requirements.txt"), "r") as file:
        required_packages = [ln.strip() for ln in file.readlines()]
    return [required_packages]


setup(
    name="Ontology Pretraining - codebase",
    version=0.1,
    author="Simone Scaboro, Beatrice Portelli",
    author_email="scaboro.simone@gmail.com",
    python_requires=">=3.8",
    packages=find_namespace_packages(),
    install_requires=requirements(),
    extras_require={
        "dev": ["black", "flake8", "isort"]
        }
)
