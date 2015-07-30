#!/usr/bin/env python
from setuptools import find_packages, setup
from setuptools.command.test import test as TestCommand
import sys


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errno = pytest.main(self.test_args)
        sys.exit(errno)


setup(
    name="lentil",
    version='0.0.0',
    url="http://siddharth.io/lentil",
    author="Siddharth Reddy",
    author_email="sgr45@cornell.edu",
    license="Apache License",
    packages=find_packages(),
    test_suite='tests',
    cmdclass={"test" : PyTest},
    description="A probabilistic model of student learning and assessment",
    long_description="\n" + open('README.md').read(),
    entry_points='''
        [console_scripts]
        lse_train=scripts.lse_train:cli
        lse_eval=scripts.lse_eval:cli
    '''
)
