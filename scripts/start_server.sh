#!/bin/bash
sudo kill -s SIGTERM $(pgrep -f 'supervisord')
cd /home/ubuntu/ARE_Miner
source environment/bin/activate
sudo supervisord -c supervisord.conf