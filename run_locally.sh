#!/bin/sh
kill -9 $(lsof -t -i:5000)
export FLASK_DEBUG=1
export FLASK_APP=./app.py && flask run
