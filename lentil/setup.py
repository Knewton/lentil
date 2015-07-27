# QUESTION This should be at the top level with the tox file

from setuptools import setup

setup(
    name='lentil-cli',
    version='0.1',
    py_modules=['train', 'evaluate'],
    install_requires=[
        'Click',
    ],
    # Alternatively, you could make one cli and have these be groups
    # You might want also to separate cli components into a subpackage so it's
    # clearer what's going on
    entry_points='''
        [console_scripts]
        lentil_train=train:cli
        lentil_eval=evaluate:cli
    ''',
)
