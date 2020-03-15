from setuptools import setup, find_packages

setup(
    name='Clearumor',
    packages=['src'],
    url='https://github.com/Institute-Web-Science-and-Technologies/CLEARumor',
    description='CLEARumor implementation',
    license = 'Apache License 2.0',
    dependency_links=['https://github.com/erikavaris/tokenizer.git'],
    python_requires='>=3.5'
)
