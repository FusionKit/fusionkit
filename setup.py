from setuptools import setup,find_packages

setup(
    name='fusionkit',
    version='0.1',
    description='FusionKit Framework',
    url='https://www.gitlab.com/gsnoep/fusionkit',
    author='Garud Snoep',
    classifiers=['Programming Language :: Python :: 3', 
                'Operating System :: OS Independent'],
    keywords='fusion simulation toolkit',
    packages=find_packages(),
    package_dir={'fusionkit':'fusionkit'},
    install_requires = ['numpy', 'scipy', 'pandas'],
)