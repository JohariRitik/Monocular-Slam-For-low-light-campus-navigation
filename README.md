To run the pipeline, you muust create a docker container for the ubuntu version 20.04.
Nvidia GPU with cuda118 drivers
Python 3.8

Test the monocular odometry for the kitti dataset and validate the results. Once done you can add the personal datasets and try and run the files in the following order
extractframes.py -----> run_mirnet_clean ------> track_mono.py

Required docker setup is added in this repo as the docker_compose.yaml file.
You might have to change the dataset location in the codes as per your system. to run a specific dataset, use the dataset address in the extract_frames.py. A folder named images_01 will be created. Use that folder to run the mirnet_clean.py to enhance the images, you will get a enhanced_0 folder with all the enhanced images
once done you can use the enhanced_0 folder to run the track_mono.py .
NOTE--- all of this is to be done inside a docker container. 
before running the track_mono.py file make sure  you open rerun application in the background to avoid any errors.

