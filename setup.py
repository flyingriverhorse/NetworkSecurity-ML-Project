'''

the setup.py file is an essential part of Python projects, especially for packaging and distribution.
It contains metadata about the project, such as its name, version, author, and dependencies.
It also specifies how to install the package and any additional files that should be included.

'''

from setuptools import setup, find_packages
# Scan through the current directory for all packages and sub-packages when there is init file in the directory.Parent folder become the package itself.
from typing import List

def get_requirements() -> List[str]:
    '''
    This function reads a requirements file and returns a list of packages.
    It removes any comments or empty lines from the file.
    '''
    requirement_lst:List[str]=[]
    # Open the requirements.txt file and read its contents
    try:
        with open('requirements.txt') as file:
            # Remove any comments or empty lines from the file
            lines=file.readlines()
            #Process each line to remove comments and empty lines
            for line in lines:
                requirement=line.strip()
                if line and not line.startswith('#'):
                    #Add the line to the list of requirements

                    #ignore empty lines and -e .
                    if requirement and requirement!= '-e .':
                        requirement_lst.append(requirement)
        # Return the list of requirements

    except FileNotFoundError:
        # If the requirements file is not found, return an empty list
        print("requirements.txt file not found.")

    return requirement_lst    

#print(get_requirements())
# Call the setup function to create the package

setup(
    name='NetworkSecurity',
    version='0.0.1',
    author='Murat Unsal',
    author_email='buzzycarl@gmail.com',
    description='A package for network security analysis and visualization',
    #long_description=open('README.md').read(),  # Read the contents of the README file for long description
    #long_description_content_type='text/markdown',
    packages=find_packages(),  # Automatically find all packages in the current directory
    install_requires=get_requirements(),  # List of dependencies from requirements.txt
)