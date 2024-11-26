#!/bin/bash
# sudo pip3 install virtualenv
cd /home/ubuntu/ARE_Miner
python3 -m venv environment
source environment/bin/activate
sudo pip install -r requirements.txt