# SET UP

You need to be on Windows for this to run.

## A. Unity
1. Unity version: `2022.1.20f1`
2. import package `x` in your Hololens project. Open scene `A` and copy all objects from there to your scene.
You should have:
- gameObject to connect to room
- gameObject send-receive 
	- get boundary by id
	- send graph get model
	- update camera position
- camera object - link the position of the camera object to the position from the tracked 3d printed camera.
3. Download the unity project from the GitHub repo.
4. open scene X.  This is the 3D visualizer.
   
## B. Python

1. install conda
2. download folder `python` from gitrepo
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
2. open a Miniconda shell and navigate to the `python` folder.
3. activate conda env `genmodel` and run `A.py`

 		conda activate genmodel
		python a.py

3. open Rhino 7
4. open Grasshopper by typing `grasshopper` then `Enter` in Rhino
5. in grasshopper: File->Open: `X.gh`

# Test
request boundary by sending id

request model by sending graph

update camera position