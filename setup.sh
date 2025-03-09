#!/bin/bash

# Upgrade pip to the latest version
python -m pip install --upgrade pip

# Install uv if not already present, then upgrade it
pip install --upgrade uv

# Install dependencies from requirements.txt using uv pip
uv pip install -r requirements.txt
