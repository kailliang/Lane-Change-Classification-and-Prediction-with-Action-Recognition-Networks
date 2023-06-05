# Lane Change Classification and Prediction with Action Recognition Networks
- Introduced framework for lane change recognition involving two action recognition approaches.
- Achieved state-of-the-art accuracy (84.79%) for lane change classification using raw RGB video data.
- Investigated temporal and spatial attention region of trained 3D models by generating Class Activation Maps.
# Paper
Liang, K., Wang, J., Bhalerao, A. (2023). Lane Change Classification and Prediction with Action Recognition Networks. In: Karlinsky, L., Michaeli, T., Nishino, K. (eds) Computer Vision – ECCV 2022 Workshops. ECCV 2022. Lecture Notes in Computer Science, vol 13801. Springer, Cham. https://doi.org/10.1007/978-3-031-25056-9_39

## Introduction
Anticipating lane change intentions of surrounding vehicles is crucial for efficient and safe driving decision making in an autonomous driving system. 
Physical variables such as driving speed, acceleration etc. can be used for lane change classification but do not contain semantic information. Deep 3D Action recognition models have been shown to be effective and can be used for lane change detection. 
This work proposes an end-to-end framework including two action recognition methods for lane change recognition, using video data collected by from front-facing cameras, with and without the need for pre-processing (vehicle bounding boxes). 

![image](https://github.com/kailliang/X3D/assets/56094206/10d7351e-9ee7-4def-bc8d-f01a8af43a07)

## Input Data
<img width="1306" alt="image" src="https://github.com/kailliang/X3D/assets/56094206/c6fa9970-200b-4c04-8ad5-2c0d304849e2">

## Methods
**RGB+3DN**: The first method utilises only the visual information collected by the front-facing cameras. This approach is tested with seven 3D action recognition models involving I3D networks, SlowFast networks, X3D networks and their variants. 
**RGB+BB+3DN**: The second method uses the same 3D action recognition networks as the first method. Bounding box information is embedded to each frame of the RGB video data to improve classification and prediction accuracy (temporal integration). This method assumes that a separate vehicle prediction method has been used on the RGB input frames.

![image](https://github.com/kailliang/X3D/assets/56094206/aabaaf10-b9a6-4e91-9198-29cc71e3ddb9)

## Results

<img width="1307" alt="image" src="https://github.com/kailliang/X3D/assets/56094206/fd9ca1cd-f9be-4668-a0f9-0c5220a8bdb2">

## Class Activation Maps
CAMs reveal that the model mainly focuses on the frames where lane changes happens, as well as the edge of the target vehicle and the lane marking which it is about to cross.

![image](https://github.com/kailliang/X3D/assets/56094206/fb2a968f-61b4-4a8a-8fb2-7d44ea977de4)
