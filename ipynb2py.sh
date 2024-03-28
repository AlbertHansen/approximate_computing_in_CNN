#!/bin/bash
echo "--- Converting .ipynb to .py ---"
find . -name "*.ipynb" -type f -exec jupyter nbconvert --to python {} \;
echo "--------------------------------"
