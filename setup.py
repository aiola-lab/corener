from setuptools import find_packages, setup


def requirements(name):
    list_requirements = []
    with open(f"{name}.txt") as f:
        for line in f:
            list_requirements.append(line.rstrip())
    return list_requirements


setup(
    name="corener",
    packages=find_packages(),
    version="0.0.1",
    description="Multi-task model for named-entity recognition, relation extraction, "
    "entity mention detection and coreference resolution",
    author="aiola",
    python_requires=">=3.7",
    install_requires=requirements("requirements"),
)
