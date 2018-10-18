from setuptools import setup, find_packages
setup(
    name = "OfflineAnomalyDetector",
    version = "0.1",
    packages = find_packages(),
    install_requires=[
        'ray>= 0.4.0',
        'thrift >= 0.11.0',
        'thriftpy >= 0.3.9',
        'scikit-learn >= 0.19.1',
        'happybase >= 1.1.0',
        'numpy >= 1.14.5',
        'pandas >= 0.20.3',
        'seaborn >= 0.8.1',
        'h5py >= 2.8.0',
        'statsmodels >= 0.8.0'
    ],
    package_data = {
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.csv'],
        # And include any *.msg files found in the 'hello' package, too:
        #'hello': ['*.msg'],
    },

)
