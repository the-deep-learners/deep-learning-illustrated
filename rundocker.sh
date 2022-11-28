#!/bin/bash
sudo docker build -t dli-stack .
sudo docker run -v $(pwd):/home/jovyan/work -it --rm -p 8888:8888 -p 6006:6006 dli-stack
