#!/bin/bash

apt update
apt install -y ffmpeg
pip install --upgrade pip
pip install decord
pip install deface
python3 -m pip install opencv-python==3.4.15.55
