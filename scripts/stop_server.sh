#!/bin/bash
sudo supervisorctl stop fastapiapp
sudo kill -s SIGTERM $(sudo supervisorctl pid)