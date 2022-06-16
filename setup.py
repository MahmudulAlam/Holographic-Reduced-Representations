import re
import setuptools

with open('HRR/__init__.py', 'r', encoding='utf-8') as fd:
    version = re.search(
        r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', fd.read(), re.MULTILINE
    ).group(1)

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(name='hrr',
                 version=version,
                 author='Mohammad Mahmudul Alam',
                 description='Holographic Reduced Representations',
                 long_description=long_description,
                 long_description_content_type='text/markdown',
                 classifiers=[
                     'Programming Language :: Python :: 3',
                     'Operating System :: OS Independent',
                 ],
                 packages=setuptools.find_packages(),
                 install_requires=[],
                 python_requires='>=3.7',
                 )
