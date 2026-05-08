To run the pipeline, you muust create a docker container for the ubuntu version 20.04.
Nvidia GPU with cuda118 drivers
Python 3.8

Test the monocular odometry for the kitti dataset and validate the results. Once done you can add the personal datasets and try and run the files in the following order
extractframes.py -----> run_mirnet_clean ------> track_mono.py

Required docker setup is added in this repo as the docker_compose.yaml file.


