#!/bin/bash
sudo supervisorctl stop fastapiapp
sudo kill -s SIGTERM $(sudo supervisorctl pid)
cd /home/ubuntu/ARE_Miner
source environment/bin/activate
sudo supervisord -c supervisord.conf