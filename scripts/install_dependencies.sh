#!/bin/bash
sudo pip3 install virtualenv
cd /home/ubuntu/ARE_Miner
virtualenv environment
source environment/bin/activate
sudo pip3 install -r requirements.txt