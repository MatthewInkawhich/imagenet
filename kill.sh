#!/bin/bash

kill $(ps aux | grep "main.py" | grep -v grep | awk '{print $2}')
kill $(ps aux | grep "main_adaptive.py" | grep -v grep | awk '{print $2}')
kill $(ps aux | grep "main_adaptive2.py" | grep -v grep | awk '{print $2}')
