#!/bin/bash
# sudo supervisorctl stop fastapiapp
sudo kill -s SIGTERM $(pgrep -f 'supervisord' | awk '{print $2}')