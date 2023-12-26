# Vision-Based Bicycle Rearview Hazard Warning System


- [Vision-Based Bicycle Rearview Hazard Warning System](#vision-based-bicycle-rearview-hazard-warning-system)
  - [Introduction \& Background](#introduction--background)
  - [An overview of our algorithm](#an-overview-of-our-algorithm)
  - [Bicycle Rearview-Based Hazard Warning System](#bicycle-rearview-based-hazard-warning-system)
  - [Lane Detection](#lane-detection)
  - [Demo video of Lane Detection](#demo-video-of-lane-detection)
  - [Object Detection](#object-detection)
  - [Object Tracking](#object-tracking)
  - [Speed \& Distance Detection](#speed--distance-detection)
    - [Estimation of vehicle distance:](#estimation-of-vehicle-distance)
    - [Estimation of vehicle speed:](#estimation-of-vehicle-speed)
  - [Safety Zone \& Vehicle motion prediction](#safety-zone--vehicle-motion-prediction)
    - [Safety Zone](#safety-zone)
    - [Vehicle motion prediction](#vehicle-motion-prediction)
  - [Hazard Prediction \& Warning](#hazard-prediction--warning)
  - [Demo 1 (YouTube Link)](#demo-1-youtube-link)
  - [Demo 2 (YouTube Link)](#demo-2-youtube-link)
___

## Introduction & Background 
There are many fully developed and practical active warning systems based on image recognition nowadays, but most of them are designed for vehicles. Bicycles that often travel on various roads are easily collided by the vehicles behind because they lack rear vision and are slower than most vehicles. Therefore, a hazard warning system is needed to reduce the occurrence of accidents. The purpose of this research is to design a hazard warning system based on the rear view of the bicycle. 
## An overview of our algorithm
Our algorithm first employs lane detection to remove uninteresting objects; then, we used Yolov4-tiny object detection algorithm to identify the type of vehicle behind and combine with DeepSort object tracking algorithm to obtain the position and speed information of the vehicle behind the cyclist. Finally, together with the logic and safety zone we designed, the rider could determine whether the rear vehicle will cause danger with the information immediately displayed on the LCD screen. 
![ss_diag](https://user-images.githubusercontent.com/69750888/207406779-1ec4d2da-66ae-4c8e-8979-9cac38399f55.jpg)
## Bicycle Rearview-Based Hazard Warning System
Rear video frames obtained by Web Cam. Jetson Nano is responsible for all computational tasks in our system. LCD and Buzzer warn the rider visually and audibly.
<div align = "center">
<img src="https://user-images.githubusercontent.com/69750888/207406770-9844ab78-a0eb-48db-a8fb-428e77f17129.png"/>
</div>

## Lane Detection
To greatly reduce the computation time of our hazard identification algorithms and for the convenience of subsequent drawing of safety zone, we should only implement image analysis on the region of interest, where is the region of the road right behind the rider that might appear the vehicles that will cause danger to the rider. Therefore, we employed lane detection to mask the region that we don’t need, outputting the part of the image that we are interested in (as shown below).
<div align = "center">
<img src ="https://user-images.githubusercontent.com/69750888/207406832-ab1d8e9d-e243-475c-8408-0e96cc22145b.png" width=400 length=400 ><img src ="https://user-images.githubusercontent.com/69750888/207406851-2224d575-ba77-4549-9386-ef7947f03584.png" width=400 length=400>
<img src ="https://user-images.githubusercontent.com/69750888/207406845-341f7791-320c-4f3d-ae62-f3beafa901d4.png" width=400 length=400 ><img src ="https://user-images.githubusercontent.com/69750888/207406847-fce61d34-3ceb-413a-b37a-232c55d0a175.png" width=400 length=400>
</div>

(Top left) Green lane lines represent all detected lane lines by lane detection and the purple lane lines are the final detected lanes obtained by fitting all green lane lines to two proper lane lines. (Top right) Utilize lane lines of each side to form a trapezoidal mask. (Bottom left) Extend the trapezoidal mask (Top right) to keeping the tracked vehicle in sight. (Bottom right) Finally, masked region that allow us to only track the vehicle that might cause danger to rider. 
## Demo video of Lane Detection
<img width="460" height="300" src="https://user-images.githubusercontent.com/69750888/207410432-6f3daeaf-e28d-4b4f-83d7-de2efd049168.mp4"/>

## Object Detection
To obtain proper distance and speed measurement of the vehicles behind, we must 
identify the type of vehicle behind and adjust the parameters of distance and speed measurement functions. 
To achieve that, we use object detection algorithms Yolov4-tiny to detect objects.
<p align="center">
<img src="https://user-images.githubusercontent.com/69750888/207414948-e81cc642-0833-4151-9017-a315bb93003d.png">
</p>

## Object Tracking
Since the object detection algorithm is to input each frame captured by the camera into the neural network for detection. Therefore, when the vehicles behind appears, Yolov4-tiny object detection algorithm can identify the class of the vehicle. However, it cannot distinguish between the identified vehicle and the newly identified vehicle and further track and predict whether the vehicles will cause dangers to the rider. Therefore, we need object tracking algorithms to track the vehicles behind and analyze the motion of the vehicles. The object tracking method we use is DeepSort. 
First, use Yolov4-tiny as a detector to detect objects and implement Kalman filter to predict the state of the detected objects and finally use Hungarian algorithm to track the objects with an ID.                                             
![deepsort](https://user-images.githubusercontent.com/69750888/207414599-5cab7103-3e9e-438d-8cfa-a3c57a2960f8.png)

## Speed & Distance Detection
The purpose of measuring the speed of the vehicles behind is to confirm whether the vehicles coming from behind will be too fast to cause dangers to the rider, as a basis for determining whether the vehicles behind the rider will cause any danger to the rider.
### Estimation of vehicle distance:
We utilized the principle of similar triangles to find the distance between the vehicles and the rider. This method requires a reference picture, knowing its actual width H and distance d, and the vehicle pixel width h can be obtained from the bounding box generated by Yolov4-tiny. By the similar triangles equation f = (d/H)*h, we can obtained the focal length of the camera, and the measured focal length is constant. Therefore, the actual distance between the vehicle and the rider d=(f*H)/h can be calculated easily (as shown below).
### Estimation of vehicle speed:
After obtaining the distance between the vehicle and the rider, we take 20 frames per second as a reference of the vehicle’s speed measurement. With the constant frame rate, we can infer the time stamps of the vehicle at certain distance. Therefore, the speed of the vehicle behind can be obtained by deriving the time difference and distance difference, combining with the state of the safety zone to determine whether the vehicle behind will cause any danger to the rider.
<p align="center">
<img src="https://user-images.githubusercontent.com/69750888/207417557-59abb9ef-023d-4dc2-88d5-692f805d66ee.png"><img src="https://user-images.githubusercontent.com/69750888/207417562-d5db31c6-910c-4c43-bf42-b85ee3edcdaa.png">
</p>

 Left. Similar triangles property to obtain the distance by single camera. Right. Illustration of width and distance relation in a frame.

## Safety Zone & Vehicle motion prediction
### Safety Zone
The position of the lane lines will be obtained through lane detection, and we will be able to draw the safety zone to determine the dangerous status of the vehicle behind. Since bicycles usually ride on the right side of the road (take the right-hand traffic country as an example) and considering that the lane lines seen by the camera will meet at a point at the far end to form a triangle. Therefore, we utilized drawing function in a very popular computer vision library – OpenCV to draw polygons, masking the uninterested region. On the right side of the lane line marked by lane detection and about half the width of the lane away from the lane line, we filled in a parallelogram with one side parallel to the lane and the other side parallel to the image (as shown below). Any vehicle behind enters in the safety zone will be considered dangerous and warn the rider. However, if the vehicles behind enter in the safety zone slowly or even the vehicles itself are static (just parked on the roadside), the system will not consider the vehicles dangerous since the low speed of those vehicle will not cause any danger to the rider.
### Vehicle motion prediction
With the tracked object obtained from DeepSort object tracking algorithm, every position of the vehicle behind in every frame is accessible. Therefore, we are able to predict the trajectory of the vehicle in the next few frames (the yellow line on the left in the Figure below) by fitting the position coordinates of the vehicles’ past trajectory into a first-degree polynomial and extend it. Thus, The future trajectory of the vehicle can thus be obtained as the pink line shown in the following Figure. 
<p align="center">
<img width=500 length=500 src="https://user-images.githubusercontent.com/69750888/207418767-c563f084-f993-4a33-991d-c7e0d258acb4.png">
</p>

Pink line shows the predicted future trajectory of the vehicle. The red and pink texts on the top left shows the current speed and the average speed of the tracked vehicle respectively and the texts on the top middle shows the danger status of the vehicle.
## Hazard Prediction & Warning
Based on the speed of the vehicle and the distance between vehicle and the rider estimated from the previous steps, we can divide the vehicle distance by the vehicle speed to predict how long it will take for the vehicle to collide with the rider. We define this time as collision time and specify three danger status – Safe, Unsafe and Dangerous – based on collision time and predicted trajectory as shown in the table below and the danger status will be displayed on the small LCD screen to warn the rider. When the danger status is Unsafe, buzzer will create a short beep to warn the rider, while buzzer will create a long beep when the danger status is dangerous (the buzzer will not create any beep at safe status).

| Status    | Collision time   |
| --------- | -----------------|
| Safe      | Longer than 1.25 seconds.                 |
| Unsafe    | Less than 1.25 seconds and the predicted motion is inside of the safety zone.            |
| Dangerous | Less than 0.75 the predicted motion is inside of the safety zone.              |

## Demo 1 (YouTube Link)
[![IMAGE ALT TEXT](http://img.youtube.com/vi/z1Axk5xVRog/0.jpg)](https://youtu.be/z1Axk5xVRog "Demo1")
## Demo 2 (YouTube Link)
[![IMAGE ALT TEXT](http://img.youtube.com/vi/UlR6IyVHxZg/0.jpg)](https://youtu.be/UlR6IyVHxZg "Demo2")

[Back to top](#vision-based-bicycle-rearview-hazard-warning-system)
