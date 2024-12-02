#!/bin/bash
# sudo supervisorctl stop fastapiapp
if pgrep -f 'supervisord'; then
    sudo kill -s SIGTERM $(pgrep -f 'supervisord')