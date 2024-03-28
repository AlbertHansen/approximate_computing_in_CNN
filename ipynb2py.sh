#!/bin/bash
echo "--- Converting SVGs to PDFs ---"
for file in **/*.ipynb; do
    jupyter nbconvert --to python $file
done
echo "-------------------------------"