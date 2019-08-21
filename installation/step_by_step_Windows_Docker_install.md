# Step-by-Step Instructions for Windows

These instructions enable you to run deep learning libraries from the comfort of interactive Jupyter notebooks. Jupyter is itself run from within a Docker container because this ensures that you'll have all of the software dependencies you need while simultaneously preventing these dependencies from clashing with the software you already have installed on your system.

Please note that **to use Docker on Windows**, you will need a 64-bit installation of Windows 10 Professional or Enterprise.

1. Install [Docker Community Edition for Windows](https://store.docker.com/editions/community/docker-ce-desktop-windows). You may need to log out and/or reboot to complete the installation.
2. Install [SourceTree](https://www.sourcetreeapp.com/).
3. Using SourceTree, clone the repository at `https://github.com/the-deep-learners/deep-learning-illustrated`. Note the directory which you used for this clone.
4. Right-click on the Docker "whale" icon in the system tray and select "Settings..." followed by "Shared Drives". Ensure that the drive which you used for the checkout is marked as shared; you will need to enter your Windows password and restart Docker at this point.
5. Start a PowerShell prompt and change into the directory where you cloned the repository.
6. Build the Docker image by executing `docker build -t dli-stack .` -- note that you'll get an error if you miss the final `.`! Also note that instead of building the Docker image, you could alternatively pull the image from Docker Hub with `docker pull jonkrohn/deep-learning-illustrated:book`
7. When that build process has finished, run the Docker container by executing `docker run -v %cd%:/home/jovyan/work -it --rm -p 8888:8888 dli-stack`. For your convenience there's also a Windows batch file, **rundocker.bat** that executes the same command, so you can simply run `rundocker.bat` from the directory where you cloned the repository.
8. In the web browser of your choice (e.g., Chrome), copy and paste the URL created by Docker (this begins with `http://localhost:8888/?token=` and should be visible near the bottom of your Terminal window)
