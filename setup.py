from setuptools import setup, find_packages

setup(
    name='ruMorpheme',
    version='0.1.3',
    author='Pavel Rykov',
    author_email='paul@drteam.rocks',
    description='Проект языковой модели для проведения морфемного анализа и сегментации слов русского языка.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/EvilFreelancer/ruMorpheme',
    packages=find_packages(),
    keywords='natural language processing, nlp, morpheme, russian',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Text Processing :: Linguistic',
        'Natural Language :: Russian',
    ],
    python_requires='>=3.11',
    install_requires=[
        "numpy~=2.1.1",
        "torch~=2.4.1",
        "huggingface-hub~=0.25.1",
    ],
)
