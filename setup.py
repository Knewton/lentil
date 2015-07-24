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
    version='0.1',
    url="http://www.knewton.com/",
    author="Siddharth Reddy",
    author_email="sgr45@cornell.edu",
    license="Apache License",
    packages=find_packages(),
    cmdclass={"test": PyTest},
    install_requires=open('requirements.txt', 'r').readlines(),
    tests_require=open('requirements.testing.txt', 'r').readlines(),
    description="A latent skill embedding model of students, assessments, and lessons",
    long_description="\n" + open('README.md').read()
)
