# SET UP

You need to be on Windows for this to run.

## A. Unity
download the unity project on the GitHub repo
open scene X.  This is the 3D visualizer

in the hololens project import package into your project
open scene A and copy all object from there to your scene

gameobject to connect to room
gameobject send-receive 
	get boundary by id
	send graph get model
	update camera postion
camera object - link the camera object to the tracked 3d printed camera

## B. Python
1. download folder `python` from gitrepo
2. install conda
3. create a new conda environment with:

	conda env create -f genmodel.yaml

4. complete package install with pip:

	pip install -r requirements.txt



## C. Rhino
1. install `Rhino 7 for Windows - One-Time Evaluation` version 90 days from here: https://www.rhino3d.com/download/
2. install lunchbox:
3. open Rhino once.

# RUN
1. run scene X from unity project in the editor
2. activate conda env `x` and run `A.py`

 		conda activate genmodel

3. open Rhino 7
4. open Grasshopper by typing `grasshopper` then `Enter` in Rhino
5. in grasshopper: File->Open: `X.gh`

# Test
request boundary
request model
update camera position