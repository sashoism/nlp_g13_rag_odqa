from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

optional_packages = {
    "tf" : ['tensorflow>=2.2.0', 'tensorflow-text', 'tensorflow-hub']
}

setup(
    name="bcqa",
    version="2.0.0",
    author="Venktesh V, Deepali Prabhu",
    author_email="venkyviswa12@gmail.com",
    description="A Benchmark for Complex Heterogeneous Question answering",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
    url="https://anonymous.4open.science/r/BCQA-05F9",
    download_url="",
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'sentence-transformers',
        'pytrec_eval',
        'faiss_cpu',
        'elasticsearch==7.9.1',
        'data',
        'toml',
        'zope.interface',
        'transformers==4.30.0',
        'protobuf',
        'openai',
        'pytrec_eval'
    ],
    extras_require = optional_packages,
    classifiers=[
        "Development Status :: Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    keywords="Information Retrieval Transformer Networks BERT PyTorch Question Answering IR NLP deep learning"
)