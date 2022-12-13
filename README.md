# Bicycle Rearview-Based Hazard Warning System using Image Analysis
## Introduction & Background 
There are many fully developed and practical active warning systems based on image recognition nowadays, but most of them are designed for vehicles. Bicycles that often travel on various roads are easily collided by the vehicles behind because they lack rear vision and are slower than most vehicles. Therefore, a hazard warning system is needed to reduce the occurrence of accidents. The purpose of this research is to design a hazard warning system based on the rear view of the bicycle. 
## An overview of our algorithm
Our algorithm first employs lane detection to remove uninteresting objects; then, we used Yolov4-tiny object detection algorithm to identify the type of vehicle behind and combine with DeepSort object tracking algorithm to obtain the position and speed information of the vehicle behind the cyclist. Finally, together with the logic and safety zone we designed, the rider could determine whether the rear vehicle will cause danger with the information immediately displayed on the LCD screen. 
![sys_diag](https://user-images.githubusercontent.com/69750888/207406779-1ec4d2da-66ae-4c8e-8979-9cac38399f55.jpg)
## Bicycle Rearview-Based Hazard Warning System
Rear video frames obtained by Web Cam. Jetson Nano is responsible for all computational tasks in our system. LCD and Buzzer warn the rider visually and audibly.
<div align = "center">
<img src ="https://user-images.githubusercontent.com/69750888/207406770-9844ab78-a0eb-48db-a8fb-428e77f17129.png">
</div>

## Lane Detection
To greatly reduce the computation time of our hazard identification algorithms and for the convenience of subsequent drawing of safety zone, we should only implement image analysis on the region of interest, where is the region of the road right behind the rider that might appear the vehicles that will cause danger to the rider. Therefore, we employed lane detection to mask the region that we don’t need, outputting the part of the image that we are interested in (as shown below).
<div align = "center">
<img src ="https://user-images.githubusercontent.com/69750888/207406832-ab1d8e9d-e243-475c-8408-0e96cc22145b.png" width=500 length=500 ><img src ="https://user-images.githubusercontent.com/69750888/207406851-2224d575-ba77-4549-9386-ef7947f03584.png" width=500 length=500>
<img src ="https://user-images.githubusercontent.com/69750888/207406845-341f7791-320c-4f3d-ae62-f3beafa901d4.png" width=500 length=500 ><img src ="https://user-images.githubusercontent.com/69750888/207406847-fce61d34-3ceb-413a-b37a-232c55d0a175.png" width=500 length=500>
</div>

(Top left) Green lane lines represent all detected lane lines by lane detection and the purple lane lines are the final detected lanes obtained by fitting all green lane lines to two proper lane lines. (Top right) Utilize lane lines of each side to form a trapezoidal mask. (Bottom left) Extend the trapezoidal mask in (Top right) to keep the tracked vehicle in sight. (Bottom right) Final masked region that allow us to only track the vehicle that might cause danger to rider. 
## Demo video of Lane Detection
<img width="460" height="300" src="https://user-images.githubusercontent.com/69750888/207410432-6f3daeaf-e28d-4b4f-83d7-de2efd049168.mp4"/>

## Object Detection
To obtain proper distance and speed measurement of the vehicles behind, we must 
identify the type of vehicle behind and adjust the parameters of distance and speed measurement functions. 
To achieve that, we use object detection algorithms Yolov4-tiny.
## Object Tracking
Since the object detection algorithm is to input each frame captured by the camera into the neural network for detection. Therefore, when the vehicles behind appears, Yolov4-tiny object detection algorithm can identify the class of the vehicle. However, it cannot distinguish between the identified vehicle and the newly identified vehicle and further track and predict whether the vehicles will cause dangers to the rider. Therefore, we need object tracking algorithms to track the vehicles behind and analyze the motion of the vehicles. The object tracking method we use is DeepSort 

## Speed & Distance Detection


## Safety Zone & Vehicle motion prediction
