try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'sipTools for stochastic inverse problems',
    'author': 'Nishant Panda',
    'url': 'https://github.com/npandachg/stochastic-inverse-problems.git',
    'author_email': 'nishant.panda@gmail.com',
    'version': '0.1',
    'packages': ['sipTools'],
    'name': 'sipTools'
}

setup(**config)
