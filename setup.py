from setuptools import setup

setup(
    name='ccnn',
    version='0.01',
    packages=['py'],
    include_package_data=True,
    install_requires=[
        'click',
        'flask',
        'jieba',
        'matplotlib',
        'mlxtend',
        'more_itertools',
        'numpy',
        'pandas',
        'sklearn',
        'tensorflow',
        'wtforms',
        'zhon',
    ],
    entry_points='''
        [console_scripts]
        cnn=py.cnn:cli
        cnn_app=py.app:cli
        ''',
)
