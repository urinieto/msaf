#!/usr/bin/env sh
python setup.py egg_info
python2 setup.py sdist bdist_wheel
python3 setup.py sdist bdist_wheel
pipreqs msaf/
mv msaf/requirements.txt .
twine upload dist/*
