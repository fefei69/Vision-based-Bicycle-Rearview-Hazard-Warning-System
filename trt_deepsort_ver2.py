"""trmet_yolo.py

This script demonstrates how to do real-time object detection with
TensorRT optimized YOLO engine.
"""

import sys 
from utils_deepsort.parser import get_config
#from tracker.tracker_tiny import Tracker_tiny --->yolo_with_plugins.py has it
from utils.yolo_with_plugins import Tracker_tiny
from utils_deepsort.draw import draw_boxes
from collections import deque
##above##revise##by##me#######################################
import os
import time
import argparse
import numpy as np 
import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver
#import RPi.GPIO as GPIO
#import I2C_LCD_driver



from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO
from utils.project_lanedetection import *
#from utils.test_buzzer import *


WINDOW_NAME = 'Trt_deepsortDemo'


def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLO model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument(
        '-c', '--category_num', type=int, default=80,
        help='number of object categories [80]')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help=('[yolov3|yolov3-tiny|yolov3-spp|yolov4|yolov4-tiny]-'
              '[{dimension}], where dimension could be a single '
              'number (e.g. 288, 416, 608) or WxH (e.g. 416x256)'))
    parser.add_argument(
        '-l', '--letter_box', action='store_true',
        help='inference with letterboxed image [False]')
    #############add deepsort yaml
    parser.add_argument('--config_deepsort', type=str, default="./configs/deep_sort.yaml")
    #parser.add_argument('--engine_path', type=str, default='./weights/yolov3_tiny_416.engine', help='set your engine file path to load')
    #######################    
    args = parser.parse_args()
    return args

def append_speed(ids,deque_list):
  speed_list = []
  for j in range(0 , len(deque_list[ids]) ):
    speed_list.append((deque_list[ids][j]))
  if len(deque_list[ids])>10:
    spd_avg = np.average(speed_list,axis=0)
    return spd_avg
  else:
    return "still appending"



  
 
    #sys.exit()
#fix bbox issues
def compute_xc_yc(out):
    w = out[:,[2]] - out[:,[0]]
    h = out[:,[3]] - out[:,[1]]
    xmin = out[:,[0]]
    ymin = out[:,[1]]
    xc = w/2 + xmin
    yc = h/2 + ymin 
    return xc,yc,w,h
    
def draw (pos,img):
    for poss in pos :
        #print("before error in draw func",poss)
        cv2.circle(img, poss, 4, (0, 255,255), -1)
        cv2.polylines(img,[np.int32(pos)], False, (0,255,255), 1)
        
        
def ez_show(img):
    img0 = np.zeros_like(img)
    cv2.line(img0,(1000,960),(586,570),(255,255,0),3)  #shift 514 pixels
    cv2.line(img0,(586,570),(500,570),(255,255,0),4)
    pol = np.array([[(224, 960), (500, 570), (586, 570),(1000, 960)]], dtype=np.int32)  
    cv2.fillPoly(img0,pol, (0,255,0))
    return img0
    
def Distance_finder(real_width, face_width_in_frame):
    '''
    This Function simply Estimates the distance between object and camera using arguments(Focal_Length, Actual_object_width, Object_width_in_the_image)
    :param1 Focal_length(float): return by the Focal_Length_Finder function

    :param2 Real_Width(int): It is Actual width of object, in real world (like My face width is = 5.7 Inches)
    :param3 object_Width_Frame(int): width of object in the image(frame in our case, using Video feed)
    :return Distance(float) : distance Estimated  

    '''
    Focal_Length = 958
    distance = (real_width * Focal_Length)/face_width_in_frame
    return distance    

def motion_cord(starting_points,line_parameters):
    slope, intercept = line_parameters
    x1 , y1 = starting_points
    y2 = y1 + 100
    #y2 = y1 + 30 #extended line
    x2 = int((y2-intercept)/(slope))
    return x1, y1, x2, y2
    
#def output_right_box (inputs,output):   
#    id = output[:,[-1]]
#    xc , yc = compute_xc_yc(inputs)
#    width = output[:,[2]] - output[:,[0]]
#    height = output[:,[3]] - output[:,[1]]
#    width = width/2
#    height = height/2
#    xmin = xc - width
#    ymin = yc - height
#    xmax = xmin + width*2
#    ymax = ymin + height*2
    #or 
#    outputs_deepsort = np.concatenate((xmin,ymin,xmax,ymax,id),axis=1)
   # print(result,"result")
   # print(outputs_deepsort,"final outpus")
#    return outputs_deepsort    
######################################
def loop_and_detect(cam, trt_yolo, tracker, conf_th,vis):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """
   # global img_final
    full_scrn = False
    fps = 0.0
    tic = time.time()
    f = [] 
    m = []
    cls = ""
    n = 0
    framenumber = -1
    unsafe_v = False
    danger_v = False
    used = False
    lanedetection = True #False
    speed = ""
    k = 0
    #tic = 0
    time_start = 0
    time_end = 0
    dis_start = 0
    dis_end = 0
    #create deque container
    pts = [deque(maxlen=30) for _ in range(100)]
    pt = [deque(maxlen=50) for _ in range(100)]
    #h_ls = [deque(maxlen=30) for _ in range(100)]
    w_list = [deque(maxlen=30) for _ in range(100)]
    car_spd = [deque(maxlen=30) for _ in range(50)]
    moto_spd = [deque(maxlen=30) for _ in range(50)]
    puttext_car = False
    puttext_moto = False
    bad = False
    avg_spd_moto = "still appending"
    avg_spd_car = "still appending"
    x_dir = []
    y_dir = []
    motion_predict = False
    drw = False
    unsafe_v = False
    danger_v = False
   #save output video
    #width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5) 
    #height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    #size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #out = cv2.VideoWriter('output_testingvid03.avi', fourcc, 10.0, (640,  480))
    #out2 = cv2.VideoWriter('combo.avi', fourcc, 20.0, (1280,960))
    out = cv2.VideoWriter('line_vis.avi', fourcc, 20.0, (1280,960))
    out1 = cv2.VideoWriter('final_res.avi', fourcc, 20.0, (1280,  960))
    out2 = cv2.VideoWriter('combo.avi', fourcc, 20.0, (1280,960))
    #out1 = cv2.VideoWriter('deepsort_out4.avi', fourcc, 20.0, (1280,  960))
   ##
    while True:
        framenumber+=1
        #mylcd = I2C_LCD_driver.lcd()
        #if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
          #  break
        img = cam.read()
        if img is None: 
       	    break
        img = cv2.resize(img, (1280, 960))
        tim = framenumber/20 
        #cv2.putText(img_better_look, f"time {tim}s",  (1100, 100), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,255,255), 2)  #bgr 
        pol = np.array([[(224, 960), (500, 570), (586, 570),(1000, 960)]], dtype=np.int32)  
        img = img.astype('uint8')
        original_image = img
        img_better_look = img
        cv2.putText(img_better_look, f"time {tim}s",  (1100, 100), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,255,255), 2)  #bgr 
        #input_cropped = frame[550:(550+IMAGE_H), 0:IMAGE_W]
        add_trans = np.zeros_like(img)
        #add_trans = add_trans[:,:,0] #force one channel 
        #img_trans = perspective_transformation(img)
        #img_trans = select_yellow_white(img_trans)
        #img_trans = canny(img_trans)
        
        '''
        lanedetection
        ==============
        filtering out unused information to speedup the system
        '''
        #yellow_white = select_yellow_white(img)
        #cannyresult = canny(yellow_white)
        #frame_for_dis = draw_dis_lines(frame)
        cannyresult = canny(img)
        #get the right vertice automatically
        vertice = get_vetices()
        cropped_image , mask = region_of_interest2(cannyresult,vertice)
        lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)  #minLineLength=40, maxLineGap=5
        #print("lines\n",lines)
        if lines is not None :
          lines = np.reshape(lines, [len(lines),4]) #lines will be None sometimes
          #avg_lines = average_slope_intercept(frame,lines)
          avg_lane, left , right = average_slope_intercept(img,lines)
          #print(len(avg_lane)) #if len(avg_lane)==1 ->only left or right if len(avg_lane)==2 ->both left and right
          #fix road disappear issue ->works well
          if len(avg_lane)==2:
            left_avg_lines = avg_lane[[0]]
            right_avg_lines = avg_lane[[1]]
            for x1 , y1 , x2 , y2 in left_avg_lines :
              xl1 , yl1 ,xl2 ,yl2 = x1 , y1 , x2 , y2  

            for x1 , y1 , x2 , y2 in right_avg_lines :
              xr1 , yr1 ,xr2 ,yr2 = x1 , y1 , x2 , y2

          elif left == True:
            for x1 , y1 , x2 , y2 in avg_lane:
              xl1 , yl1 ,xl2 ,yl2 = x1 , y1 , x2 , y2

          elif right == True:
            for x1 , y1 , x2 , y2 in avg_lane:
              xr1 , yr1 ,xr2 ,yr2 = x1 , y1 , x2 , y2

          try:   
            #vertices_polly = np.array([[(xl1, yl1), (xl2, yl2), (xr2, yr2), (xr1, yr1)]], dtype=np.int32)
            if xr1 - xl1 < 900:
              xr1 = xl1 + 1100
            if (xr2+5)-(xl2-5) < 130:
              xl2 = xr2 + 10 +150
            if (xr2+5) < (xl2-5) :
              xl2 , xr2 = xr2 + 10 , xl2 - 10 
            vertices_polly = np.array([[(xl1, yl1), (xl2-5, yl2-80), (xr2+5, yr2-80), (xr1, yr1)]], dtype=np.int32) #extend trapezoid
            vertices_polly_unextd = np.array([[(xl1, yl1), (xl2, yl2), (xr2, yr2), (xr1, yr1)]], dtype=np.int32) #unextend trapezoid
          except (NameError,OverflowError):
            print("xl1 is not defined ->only one side of line works")
            
        else:
          print("default avg_lines(not detecting lanes)")
          avg_lane = np.array([[0 ,572 ,479 ,205],   #0 572 ; 479  205 ; 641 193 ; 1268 481

                             [1268 ,481 ,641 ,193]])
          vertices_polly = None
        
        img_zero = np.zeros_like(img)
        img0 =np.zeros_like(img) 
        color_polly =  (0,255,0) #BGR
        line_image_not_avg = draw_lines1(img0, lines)
        line_image = draw_lines2(img, avg_lane)
        line_visualize = cv2.addWeighted(img_better_look,1,line_image,1,1)
        line_visualize = cv2.addWeighted(line_image_not_avg,1,line_visualize,1,1)
        god = filterout2(img,vertice,mask)
        #print(vertices_polly)
        try:
          #cv2.fillPoly(line_image, vertices_polly, color_polly)
          cv2.fillPoly(img_zero, vertices_polly_unextd, (0,255,0))
          filtered = filterout(img,vertices_polly)
          #print("vertices polly :\n",vertices_polly)
          if lanedetection == True:
             
            img=filtered #img = filtered
            #img = god
            #img = img   #not filtering
          #god = filterout2(frame,mask)
        except NameError:
          #filtered = original_image 
          filtered = god
          print("vertices polly is not defined")
          
        normal_result = cv2.addWeighted(img,1,img_zero,1,1)
        #print("vertices polly :",vertices_polly)
        combo_image = cv2.addWeighted(img, 1, line_image, 1, 1)
        #combo_image = cv2.addWeighted(combo_image, 1, line_image_not_avg, 1, 1) #addWeighted function cant add two srcs
        img_notavg = cv2.addWeighted(img, 1, line_image_not_avg, 1, 1)
                                  
        #allowing safety zone to draw on ->or the color of safety zone will be too dark        
        im0 = np.zeros_like(img)      
        if unsafe_v == False and danger_v == False:
          cv2.putText(img_better_look, f"safe", (800, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)  #bgr
          
          #img0 = np.zeros_like(img_better_look)
          cv2.fillPoly(im0,pol, (0,255,0))
          #img_better_look = cv2.addWeighted(img0,0.7,img_better_look,1,1)
             
          #speed estimate zone
          #cv2.line(img_better_look,(50,530),(1260,530),(0,127,255),3)
          #cv2.line(img_better_look,(50,560),(1260,560),(0,127,255),3)
        
        '''
        yolov4 + Tensorrt
        
        '''        
        
        boxes, confs, clss = trt_yolo.detect(img, conf_th)
        #yolo_init = boxes
        #img0 = ez_show(img)
        #img0 = cv2.addWeighted(img0,0.7,img,1,1)
    
        '''
        object tracking by DeepSort
        
        '''
        
        #compute width and height of bboxs
        output = boxes
        w = output[:,[2]] - output[:,[0]]
        h = output[:,[3]] - output[:,[1]]
        xc , yc , w , h = compute_xc_yc(output)
        #print(xc,yc,"center")
        boxes = np.concatenate((xc,yc,w,h),axis=1)
        outputs = tracker.run(img, boxes, confs)

        #print('boxes_changed\n',boxes,'confs\n',confs,'clss',clss,"\n############")
        #print("         deepsort bboxs:            \n ",outputs)
        #for tensorrt_yolo
        #img_better_look = vis.draw_bboxes(img_better_look, output, confs, clss) 
        img_better_look = vis.draw_bboxes(img_better_look, output, confs, clss) 
        if len(clss)==1:
          clss = int(clss)
          cls = vis.cls_dict.get(clss)
        else:
          cls = ""
          #print("class      :",cls)
        #print("the type of class :\n",type(clss),"class :",clss)
        #f = []
        
        
        '''
        safety zone geometry setting
        
        '''
        
        
        if len(outputs) > 0 :
            #print(xc,"xc")
            #outputs.astype(int)
            #print("before x1 y1......",outputs)
            for x1,y1,x2,y2,ids in outputs:
              xmin = x1
              ymin = y2
              w = x2 - x1 #w
              h = y2 - y1 #h
              xc = w/2 + x1 #xmin = x1
              yc = h/2 + y1 #ymin = y2
              xc = int(xc)
              yc = int(yc)
              w = int(w)
              h = int(h)
              low_mid = (xc,y2)
              low_left = (x1,y2)
              center = (xc,yc,h)
              #cent = (xc,yc)
              w_tim = (w,tim,cls)
              #xc = np.array(xc[0,0],dtype = np.int32) 
              #yc = np.array(yc[0,0],dtype = np.int32)
              x_res_yc = 1.062*yc - 20 
              x_res_y = 1.062*y2 - 20
              pol = np.array([[(224, 960), (500, 570), (586, 570),(1000, 960)]], dtype=np.int32)  
              #pts[ids].append(center)
              print("id",ids,"w_tim",w_tim)
              pt[ids].append(low_left)
              #h_ls[ids].append(h)
              w_list[ids].append(w_tim)
              #print("pt:\n",pt,"\n")
              print("w_list:\n",w_list,"\n")
              
              #print("the ids now :",ids,"\n")
              for j in range(0, len(pt[ids])): #start with 1
                #cent = (pts[ids][j][0] , pts[ids][j][1])      
                #cent = (pts[ids][j-1] , pts[ids][j])
                
                
                #cv2.line(img_better_look,(pt[ids][j-1]) , (pt[ids][j]),(0,255,255),3)
                #greatest > curr
                #print("len(pt[ids])",len(pt[ids]))
                if abs(pt[ids][j][1] - pt[ids][j-1][1]) < 10 :
                  #print("in abs!!!!!!",(pt[ids][j-1]) , (pt[ids][j]))
                  cv2.line(img_better_look,(pt[ids][j-1]) , (pt[ids][j]),(0,255,255),3)
                if len(pt[ids]) > 5:
                  if j%5 == 0:
                    #motion = True
                    #x_dir_avg = np.average(x_dir)
                    #y_dir_avg = np.average(y_dir)
                    #parameters_avg = np.polyfit(x_dir_avg, y_dir_avg, 1)
                    #cv2.line(img_better_look,(pts[ids][j-1][0],pts[ids][j-1][1]),(pts[ids][j][0],pts[ids][j][1]),(255,255,255),3)
                    x_dirr = (pt[ids][0][0] , pt[ids][j][0])
                    y_dirr = (pt[ids][0][1] , pt[ids][j][1])
                    #x direction has same value
                    #if pt[ids][0][0] == pt[ids][j][0] :
                      #x_dirr = (pt[ids][0][0] , pt[ids][0][0]+2)
                      #x direction has same value
                    if pt[ids][0][1] == pt[ids][j][1] :
                      y_dirr = (pt[ids][0][1] , pt[ids][0][1]+5)
                      
                    parameters = np.polyfit(x_dirr, y_dirr, 1)
                    drw = True
                    #parameters = np.polyfit(x_dir, y_dir, 1)
                    print("x dirr y dirr",x_dirr,y_dirr)
              if drw == True:
                x1 ,y1 ,x2 ,y2 = motion_cord((x1,y2) , parameters)
                #print("x1 y1 x2 y2",parameters,x1,y1,x2,y2)
                x_res_y2 = 1.062*y2 - 20
                cv2.line(img_better_look,(x1,y1),(x2,y2),(255,0,255),3)
                drw = False
                if x_res_y2 > x2 and y2 > 570 :
                  motion_predict = True
                  #cv2.putText(img_better_look, f"motion True", (700, 80), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0,255,255), 2)  #bgr
                  #parameters = np.polyfit(x_dir, y_dir, 1)
                  #parameters = np.polyfit((pts[ids][j][0]), (pts[ids][j][1]),1)
                  #print("(pts[ids][j][0] :",(pts[ids][j][0]))
                  
                  #x1 ,y1 ,x2 ,y2 = motion_cord((pts[ids][j][0], pts[ids][j][1] + (pts[ids][j][2]//2)) , parameters)
                  #cv2.line(add_trans,(x1,y1),(x2,y2),(255,0,255),1)
                  #cv2.line(img_better_look,(x1,y1),(x2,y2),(255,0,255),3)
                  
                  
                #trans = perspective_transformation(add_trans)
                #x1,y1 = pts[ids][j-1]
                #x2,y2 = pts[ids][j]
                #cv2.line(trans,(x1,(y1*300//960)), (x2,(y2*300//960)),(0,255,255),3)
                #img_trans = cv2.addWeighted(img_trans,1,trans,1,1) 
                
                
              
              
                
              '''
              speed estimation 
              ==============
              estimate speed using deque method
              '''
              for i in range(0, len(w_list[ids])):
                #print("w_list workssssssssssssssssss")
                #print("len(w_list[ids]) :",len(w_list[ids]),"; i :",i)
                #print("k in for loop",k)
                #i+=2
                width_curr = w_list[ids][i][0]
                if i%5 ==0 and k+1 <= i and len(w_list[ids]) > 5:  #sample every 3 points
                  #print("k in if statement",k)
                  width_1 = (w_list[ids][k-1][0])  #near wider
                  width_2 = (w_list[ids][k-5][0])  #far
                  #width_curr = (w_list[ids][i][0])
                  time_passed = abs((w_list[ids][k-1][1]) - (w_list[ids][k-5][1]))
                  name = (w_list[ids][k-1][2])
                  if name == "" :
                    name = (w_list[ids][k-5][2])
                  print("time passed :",time_passed," time1 ",(w_list[ids][k-1][1])," time2 ",(w_list[ids][k-5][1]),)
                  print("width difference :",abs(width_1-width_2))
                  if time_passed > 0:
                    if name == "car":
                      dis_car2 = Distance_finder(210,width_2)/100
                      dis_car1 = Distance_finder(210,width_1)/100
                      dis_car =  Distance_finder(210,width_curr)/100
                      dis_diff_car = abs(dis_car2 - dis_car1)
                      car_speed = (dis_diff_car/time_passed)*3600/1000
                      car_spd[ids].append(car_speed)
                      avg_spd_car = append_speed(ids,car_spd)
                      if avg_spd_car != "still appending":
                        #print(avg_spd_car)
                        avg_spd_car = int(avg_spd_car)
                        puttext_car = True
                        #cv2.putText(img_better_look, f"average car speed {avg_spd}   km/h",  (50, 150), cv2.FONT_HERSHEY_COMPLEX, 0.65, (0,0,255), 2)  #bgr
                      #print("car speed",car_spd)
                      #print("car speed cord  :",car_spd[ids],"average speed :",avg_spd_car)
                      #cv2.putText(img_better_look, f"car speed {int(car_speed)}   km/h",  (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.65, (255,0,255), 2)  #bgr
                      #no class found ->usually motorbike
                    if name == "motorbike" or name =="":
                      dis_moto1 = Distance_finder(85,width_1)/100
                      dis_moto2 = Distance_finder(85,width_2)/100
                      dis_moto = Distance_finder(85,width_curr)/100
                      dis_diff_moto = abs(dis_moto2 - dis_moto1)
                      moto_speed = (dis_diff_moto/time_passed)*3600/1000
                      moto_spd[ids].append(moto_speed)
                      avg_spd_moto = append_speed(ids,moto_spd)
                      #cv2.putText(img_better_look, f"motorbike speed {int(moto_speed)}  km/h",  (50, 90), cv2.FONT_HERSHEY_COMPLEX, 0.65, (255,255,0), 2)  #bgr    
                      if avg_spd_moto != "still appending":
                       # print(avg_spd_moto)
                        avg_spd = int(avg_spd_moto)
                        puttext_moto = True 
                    
                #when w_list is short   
                elif len(w_list[ids]) == 5:
                  width_1 = (w_list[ids][4][0])  #near wider
                  width_2 = (w_list[ids][0][0])  #far
                  time_passed = abs((w_list[ids][4][1]) - (w_list[ids][0][1]))
                  name = (w_list[ids][3][2])
                  if name == "" :
                    name = (w_list[ids][1][2])
                  if name == "car":
                    dis_car2 = Distance_finder(210,width_2)/100
                    dis_car1 = Distance_finder(210,width_1)/100
                    dis_diff_car = abs(dis_car2 - dis_car1)
                    car_speed = (dis_diff_car/time_passed)*3600/1000
                    car_spd[ids].append(car_speed)
                    avg_spd_car = append_speed(ids,car_spd)
                    if avg_spd_car != "still appending":
                      print(avg_spd_car)
                      avg_spd_car = int(avg_spd_car)
                      puttext_car = True
                  if name == "motorbike":
                    dis_moto1 = Distance_finder(85,width_1)/100
                    dis_moto2 = Distance_finder(85,width_2)/100
                    dis_diff_moto = abs(dis_moto2 - dis_moto1)
                    moto_speed = (dis_diff_moto/time_passed)*3600/1000
                    moto_spd[ids].append(moto_speed)
                    avg_spd_moto = append_speed(ids,moto_spd)
                    if avg_spd_moto != "still appending":
                      avg_spd_moto = int(avg_spd_moto)
                      puttext_moto = True      
                #k = 3*i
                k = i
                '''
                plot speed information
                
                '''
              if puttext_car == True and  avg_spd_car != "still appending" and car_speed != 0 and avg_spd_car != 0:
                car_imptim_avg = round(dis_car/(avg_spd_car*1000/3600),2)               
                car_imptim = round(dis_car/(car_speed*1000/3600),2)
                cv2.putText(img_better_look, f"average car speed {avg_spd_car} km/h Collision time {car_imptim_avg} s",  (50, 90), cv2.FONT_HERSHEY_COMPLEX, 0.65, (0,0,255), 2)  #bgr
                cv2.putText(img_better_look, f"car speed {int(car_speed)} km/h  Collision time {car_imptim}s  ",  (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.65, (255,0,255), 2)  #bgr
                if car_imptim_avg < 1.25 and motion_predict == True:
                  unsafe_v = True
                  danger_v = False
                  cv2.putText(img_better_look, f"Unsafe", (700, 50), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,0,255), 2)  #bgr
                  cv2.fillPoly(im0,pol, (255,0,255))
                if car_imptim_avg < 0.75 and motion_predict == True:
                  danger_v = True
                  unsafe_v = False
                else:
                  danger_v = False
              if puttext_moto == True and  avg_spd_moto != "still appending" and moto_speed !=0 and avg_spd_moto !=0:
                moto_imptim = round(dis_moto/(moto_speed*1000/3600),2)
                moto_imptim_avg = round(dis_moto/(avg_spd_moto*1000/3600),2)
                cv2.putText(img_better_look, f"motorbike speed {int(moto_speed)} km/h Collision time {int(moto_imptim)} s",  (50, 140), cv2.FONT_HERSHEY_COMPLEX, 0.65, (255,255,0), 2)  #bgr    
                cv2.putText(img_better_look, f"avg motorbike speed {int(avg_spd_moto)} km/h Collision time {int(moto_imptim_avg)} s",  (50, 180), cv2.FONT_HERSHEY_COMPLEX, 0.65, (255,255,0), 2)  #bgr 
                if moto_imptim_avg < 1.25 and motion_predict == True:
                  unsafe_v = True
                  danger_v = False
                if moto_imptim_avg < 0.75 and motion_predict == True:
                  danger_v = True
                  unsafe_v = False
                else:
                  danger_v = False
              
                cv2.putText(img_better_look, f"danger True", (700, 80), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,255,255), 2)  #bgr
         #speed estimation ends       
                
                
              
              if cls == "car":
                dis = Distance_finder(210,w)//100
                dis = int(dis)
                cv2.putText(img_better_look, f"Distance {dis} m", (xc-6, yc-6), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,255), 2)  #bgr
                #560
                if yc > 530 and yc <540:
                  time_start = tim
                  dis_start = dis
                #
                if yc > 560 and yc < 570:
                  used = True
                  time_end = tim
                  dis_end = dis
                  if used == True:
                    if time_end-time_start == 0:
                      print("time = 0") 
                    else:
                      speed = ((dis_start-dis_end)/(time_end-time_start))*3600/1000
                      
                      #cv2.putText(img_better_look, f"distance start {dis_start}  distance end {dis_end}",  (100, 400), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)  #bgr
                      if speed < 0:
                        speed = "calculating speed"
                      else:
                        bad = True
                        speed = int(speed)
                if bad ==True :
                  print("")
                  #cv2.putText(img_better_look, f"car speed {speed} km/hr ",  (50,400), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,255,255), 2)
                      
                
                      #cv2.putText(img_better_look, f"distance start {dis_start}  distance end {dis_end}",  (100, 400), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)  #bgr
                      #cv2.putText(img_better_look, f"time start {time_start}  time end {time_end}",  (100, 500), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)  #bgr
                      #cv2.putText(img_better_look, f"car speed {speed} km/hr ",  (50,400 ), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,255,255), 2)
                #f.insert(0,(xc,yc))
                #print("car center lsit :  ",f)
                
                
              if cls == "motorbike":
                 dis = Distance_finder(85,w)//100
                 dis = int(dis)
                 print("distance",dis)
                 cv2.putText(img_better_look, f"Distance {dis} m", (xc-6, yc-6), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,255), 2)  #bgr 
                 if yc > 530 and yc <540:
                   time_start = tim
                   dis_start = dis                           
                 if yc > 560 and yc < 570:
                   used = True
                   time_end = tim
                   dis_end = dis
                   if used == True:
                     if time_end-time_start == 0:
                       print("time = 0")
                     else: 
                       bad = True
                       speed = ((dis_start-dis_end)/(time_end-time_start))*3600/1000
                     #cv2.putText(img_better_look, f"distance start {dis_start}  distance end {dis_end}",  (100, 400), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)  #bgr
                 if bad == True :
                   print("debuging speed",speed)
                   if speed < 0 :
                     print("speed is negative")
                   else :
                     speed = int(speed)
                     cv2.putText(img_better_look, f"motorcycle speed {speed} km/hr ",  (50,300 ), cv2.FONT_HERSHEY_COMPLEX, 0.65, (0,255,255), 2)   
                           
                         #cv2.putText(img_better_look, f"distance start {dis_start}  distance end {dis_end}",  (100, 400), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)  #bgr                           
                         #cv2.putText(img_better_look, f"time start {time_start}  time end {time_end}",  (100, 500), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)  #bgr
                    #cv2.putText(img_better_look, f"motorcycle speed {speed} km/hr ",  (50,300 ), cv2.FONT_HERSHEY_COMPLEX, 0.65, (0,255,255), 2)   
                   #m.insert(0,(xc,yc))
                   #print("motorbike center list:  ",m)
                   
              '''
              safety zone
              =============
              safe unsafe danger 
              '''
              
              #unsafe
              
              #if x_res_y > xmin and ymin > 570 and danger==False or unsafe_v == True and danger==False:
              if unsafe_v == True: #and danger==False:
                #buzz(unsafe_v)
                #mylcd.lcd_display_string("unsafe",  2,3)
                #unsafe = True
                print("satis1")
                cv2.putText(img_better_look, f"Unsafe", (700, 50), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,0,255), 2)  #bgr
                #img1 = np.zeros_like(img_better_look)
                cv2.fillPoly(im0,pol, (255,0,255))
                #img = cv2.addWeighted(img1,0.7,img,1,1)
              '''             
              if ymin > 570 and xmin < 586 and danger == False: #straight behind
                unsafe = True
                print("satis2")                                   
                cv2.putText(img_better_look, f"Unsafe", (700, 50), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,0,255), 2)  #bgr
                #img2 = np.zeros_like(img_better_look)
                cv2.fillPoly(im0,pol, (255,0,255))
                #img = cv2.addWeighted(img2,0.7,img,1,1)
              '''
              #dangerous         
              #if x_res_yc > xc and yc > 570 or danger_v == True and unsafe == False:
              if danger_v == True: #and unsafe == False:
                #mylcd.lcd_display_string("dangerous!",  2,3)
                #buzz(unsafe_v)
                print("satis3")
                #danger = True #dangerous
                #unsafe = False
                cv2.putText(img_better_look, f"dangerous!!", (800, 50), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,0,255), 2)  #bgr
                #img3 = np.zeros_like(img_better_look)
                cv2.fillPoly(im0,pol, (0,0,255))
                #img = cv2.addWeighted(img3,0.7,img,1,1)
              '''
              elif yc > 570 and xc < 586:
                danger = True
                unsafe = False
                print("satis4")
                cv2.putText(img_better_look, f" danger!!", (800, 50), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,0,255), 2)  #bgr                          
                #img4 = np.zeros_like(img_better_look)
                cv2.fillPoly(im0,pol, (0,0,255))
                #img = cv2.addWeighted(img4,0.7,img,1,1)
               '''
            #f.insert(0,(int(xc),int(yc)))
            #f.astype(numpy.int64)
            #print(f,"ffffffffffffffffff")
            #draw(f,img)
            #cv2.circle(img,(int(xc),int(yc)),4,(0,255,0),-1)
              #if len(f) > 1:
                #cv2.circle(img,f,4,(0,255,255),-1)
            #n+=1
            #f.insert(0,yc)
            #print(f,"center points list for y")

        #for tensorrt_yolo
        #print(clss)
        #img = show_fps(img, fps)
        ###################################
        #outputs = tracker.run(img,boxes, confs)
	# draw boxes for visualization -deepsort 
        
        #   
      #  outputs = output_right_box(outputs) there will be a error because deepsort has not updated yet len(outputs) = 0 
        #bbox_xyxy = output[:, :4]
        #identities = output[:, -1]
        
        
        if len(outputs) > 0:
            #print("outputs after output right box",outputs)
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -1]
            img_final = draw_boxes(img_better_look, bbox_xyxy, identities)
            #img_better_look = show_fps(img_better_look, fps)
        ###################################
        img_better_look = show_fps(img_better_look, fps)
        imgx = img0
        real_result = cv2.addWeighted(img_better_look,0.7,img,1,1) #view lanedetection filtering
        img_better_look = cv2.addWeighted(im0,1,img_better_look,1,1) # 
        
        out.write(line_visualize)
        out1.write(img_better_look)
        out2.write(combo_image)
        
        #show result
        #cv2.imshow(WINDOW_NAME, img)
        ####
        #cv2.imshow("normal lanedetection without extended",normal_result)
        #cv2.imshow("combo img",combo_image)
        #cv2.imshow(" img",img_notavg) 
        #cv2.imshow("avg line",line_visualize)
        ####
        #cv2.imshow("only safety zone",img_better_look)
       # cv2.imshow("cropped image ",cropped_image)
        #cv2.imshow("real result ",real_result)
        #cv2.imshow("example ",real_result)
        #cv2.imshow("image better look",img_better_look)
        #cv2.imshow("predict motion  ",img_trans)
       # cv2.imshow("predict motion  ",god)
       # try:
        #  cv2.imshow("filtered",filtered)
       # except NameError:
       #   print("")
        #cv2.imshow("imgx",imgx)
        toc = time.time()
        curr_fps = 1.0 / (toc - tic)
        # calculate an exponentially decaying average of fps number
        fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
        tic = toc
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)


def main():
    args = parse_args()
    ########
    cfg = get_config()
    cfg.merge_from_file(args.config_deepsort)    
    ########
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)

    cam = Camera(args)
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')
    ########    
    tracker = Tracker_tiny(cfg) 
    ########
    cls_dict = get_cls_dict(args.category_num)
    yolo_dim = args.model.split('-')[-1]
    if 'x' in yolo_dim:
        dim_split = yolo_dim.split('x')
        if len(dim_split) != 2:
            raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)
        w, h = int(dim_split[0]), int(dim_split[1])
    else:
        h = w = int(yolo_dim)
    if h % 32 != 0 or w % 32 != 0:
        raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)

    trt_yolo = TrtYOLO(args.model, (h, w), args.category_num, args.letter_box)

    #open_window(WINDOW_NAME, 'Camera TensorRT YOLO Demo',cam.img_width, cam.img_height)
        
    vis = BBoxVisualization(cls_dict)
    
    loop_and_detect(cam, trt_yolo, tracker, conf_th=0.3, vis=vis)

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
   main()
