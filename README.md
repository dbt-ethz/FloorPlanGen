# SET UP

You need to be on Windows for this to run.

## A. Unity
1. Unity version: `2022.1.20f1`, please switch to the one that is recommended for Hololens `2020.3.40f1`.
2. Go to https://assetstore.unity.com, log in to your account and add `PUN 2 FREE` (https://assetstore.unity.com/packages/tools/network/pun-2-free-119922) to your assets.
3. In the Unity Editor with your project open, open the Package Manager and switch to 'My Assets'. Find `PUN 2 FREE` and download it. Then import it.
4. Download the unity project from the GitHub repo.
5. open scene `client-server`.  This is the 3D visualizer.
6. Hololens = client, laptop = server

7. In the SendReceive.cs you have the framework for the following:
   1. Client2Server: get boundary by id (1,2 or 3) - the boundary json string will be in `GameSettingsSingleton.Instance.boundaryJsonString = jsonString;`
   2. Client2Server: send the graph - the client will save the graph, the server will save it as graph.json in its Resources folder
   3. Server2Client: send mesh to client


8. update camera position: The GameObject TrackedCamera has an Id and an observer (Photon Transfrom View). So in the client link your tracked 3d printed camera object to this Tracked Camera. It will be mirrored in the server.



   
## B. Python

1. install miniconda in case you don't have it: https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html
2. update conda by navigate to the anaconda directory in the Miniconda shell and Run `conda update conda`
3. download folder `python` from gitrepo
4. create a new conda environment with:

		conda env create -f genmodel.yaml

5. complete package install with pip:

	pip install -r requirements.txt



## C. Rhino
1. install `Rhino 7 for Windows - One-Time Evaluation` version 90 days from here: https://www.rhino3d.com/download/
2. install lunchbox as follows. Open Rhino 7. Go to Tools->Package Manager...
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