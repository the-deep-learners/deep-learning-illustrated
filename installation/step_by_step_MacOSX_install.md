# Step-by-Step Instructions for macOS

These instructions enable you to run TensorFlow code from the comfort of interactive Jupyter notebooks. Jupyter is itself run from within a Docker container because this ensures that you'll have all of the software dependencies you need while simultaneously preventing these dependencies from clashing with the software you already have installed on your system.

## Install

1. Open the Terminal application ([like this](http://www.wikihow.com/Open-a-Terminal-Window-in-Mac))
2. To install in your home directory (this is my recommended default):
	* Type `cd ~` into the command-line prompt and
	* *Execute* the line by pressing the **return** key on your keyboard
3. Retrieve all of the code for this LiveLessons by executing `git clone https://github.com/the-deep-learners/deep-learning-illustrated` (if you haven't used `git` before, you may be prompted to install Xcode -- do it!)
4. [Install the Docker "Stable channel"](https://docs.docker.com/docker-for-mac/install/) (if you are already using an older version of Docker and run into installation issues downstream, try updating to the latest version of Docker)
5. Start Docker, e.g., by using Finder to navigate to your Applications folder and double-clicking on the Docker icon
6. Back in Terminal, execute `source deep-learning-illustrated/installation/let_jovyan_write.sh` so that you can write to files in the *deep-learning-illustrated* directory from inside the Docker container we'll be running shortly
7. Move into the *deep-learning-illustrated* directory by executing `cd deep-learning-illustrated`
8. Build the Docker image by executing `sudo docker build -t dli-stack .` -- note that you'll get an error if you miss the final `.` in the command! Also note that instead of building the Docker image, you could alternatively pull the image from Docker Hub with `docker pull jonkrohn/deep-learning-illustrated:book`
9. Run the Docker image as a Docker container by executing `sudo docker run -v $(pwd):/home/jovyan/work -it --rm -p 8888:8888 dli-stack` (you can think of the image as a recipe and the container as the cake produced by the recipe). For your convenience there is a bash script, **rundocker.sh** that executes the same command, so you can simply run `source rundocker.sh`. This command must be executed from the directory where you cloned the repository
10. In the web browser of your choice (e.g., Chrome), copy and paste the URL created by Docker (this begins with `http://localhost:8888/?token=` and should be visible near the bottom of your Terminal window)

## Shutdown

You can shutdown the Jupyter notebook by returning to the Terminal session that is running it and hitting the **control** and **c** keys simultaneously on your keyboard.

## Restart

You can restart the Jupyter notebook later by following steps nine and ten alone.

