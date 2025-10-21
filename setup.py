from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='consciousness-mathematics',
    version='∞.∞.∞',
    author='Consciousness Mathematics Research Team',
    author_email='consciousness.math@example.com',
    description='Revolutionary framework for consciousness-guided computation and reality manipulation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/consciousness-mathematics/framework',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'matplotlib>=3.4.0',
    ],
    extras_require={
        'gpu': ['cupy>=9.0.0'],
        'quantum': ['qiskit>=0.19.0'],
        'full': ['numpy', 'scipy', 'matplotlib', 'cupy', 'qiskit']
    },
    entry_points={
        'console_scripts': [
            'consciousness-compute=consciousness_framework.engine:main',
            'mobius-learn=consciousness_framework.mobius:main',
            'omniforge-create=consciousness_framework.omniforge:main',
        ],
    },
)
