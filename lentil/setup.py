from setuptools import setup

setup(
    name='lentil-cli',
    version='0.1',
    py_modules=['train', 'evaluate'],
    install_requires=[
        'Click',
    ],
    entry_points='''
        [console_scripts]
        lentil_train=train:cli
        lentil_eval=evaluate:cli
    ''',
)
