import sys


def check_venv(venv='mt_env'):
    if sys.prefix.split('/')[-1] != venv:
        raise ConnectionError('Not connected to correct virtual environment')


check_venv()
