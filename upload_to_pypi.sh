#!/usr/bin/env sh
python setup.py egg_info
python2 setup.py sdist bdist_wheel
python3 setup.py sdist bdist_wheel
twine upload dist/*
cd docs
make html
cd _build/html
zip -r msaf-html.zip *
