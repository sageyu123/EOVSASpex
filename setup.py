from setuptools import setup

setup(name='EOVSA-spex', version='0.0.1', packages=['EoSpex'], install_requires=['PyQt5', 'sunpy', 'astropy'],
    package_data={'': ['QSS/*.qss', 'resource/*.jpg', 'resource/*.ascii', 'resource/*.ico', 'widgets/*.py'], },
    entry_points={'console_scripts': ['eospex = EoSpex.eospex:start'], },

    license='xxx', author='sjyu1988', author_email='sjyu1988@gmail.com', url='empty', description='Spectra anlysis tool for EOVSA',
    keywords=['Astronomy', 'Solar', 'Radio', 'EOVSA'], )
