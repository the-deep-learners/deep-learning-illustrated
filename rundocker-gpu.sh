#!/bin/bash
sudo nvidia-docker run -v $(pwd):/home/jovyan/work -it --rm -p 8888:8888 dli-gpu-stack
