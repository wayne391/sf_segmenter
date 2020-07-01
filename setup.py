from setuptools import setup, find_packages

setup(
    name='sf_segmenter',
    version='0.0.1',
    description='',
    author='wayne391',
    author_email='s101062219@gmail.com',
    url='https://github.com/wayne391/sf_segmenter',
    packages=find_packages(),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
    ],
    keywords='music audio midi mir',
    license='MIT',
    install_requires=[
        'miditoolkit >= 0.1.14',
    ]
)


"""
python setup.py sdist
twine upload dist/*
"""