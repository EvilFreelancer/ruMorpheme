from setuptools import setup, find_packages

setup(
    name='ruMorpheme',
    version='0.1.1',
    author='Pavel Rykov',
    author_email='paul@drteam.rocks',
    description='Проект языковой модели для проведения морфемного анализа и сегментации слов русского языка.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/EvilFreelancer/ruMorpheme',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
    install_requires=[
        "numpy~=2.1.1",
        "torch~=2.4.1",
        "huggingface-hub~=0.25.1",
    ],
)
