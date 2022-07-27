__author__ = "Olli Knuuttila"
__date__ = "$Nov 25, 2016 8:05:11 PM$"

from setuptools import setup, find_packages

setup(
    name='visnav',
    version='0.1',
    packages=find_packages(include=['visnav*']),
    include_package_data=True,
    package_data={'visnav.render': ['*.frag', '*.vert', '*.geom']},

    # Declare your packages' dependencies here, for eg:
    install_requires=['numpy', 'scipy', 'numba', 'py-opencv', 'quaternion'],
    # optional_packages=['moderngl', 'objloader', 'astropy'],

    author=__author__,
    author_email='olli.knuuttila@gmail.com',

    summary='Visual navigation prototyping',
    url='https://github.com/oknuutti/hw_visnav',
    license='MIT',
)