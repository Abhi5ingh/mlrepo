from setuptools import find_packages, setup

def get_requirements(filename:str)->list:
    requirements=[]
    with open(filename, 'r') as file:
        requirements = file.read().splitlines()
        if "-e ." in requirements:
            requirements.remove("-e . ")
    return requirements

setup(
    name='mlrepo_proj',
    version='0.0.1',
    author='Abhi5ingh',
    author_email='abhisteak@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)