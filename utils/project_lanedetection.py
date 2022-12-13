import numpy as np 
import cv2
from numpy.lib.function_base import append



def select_yellow_white(img_org):
    #hsv_img = cv2.cvtColor(img_org, cv2.COLOR_RGB2HLS)
    hsv_img = cv2.cvtColor(img_org, cv2.COLOR_BGR2HLS)
    
    img_hsv = cv2.cvtColor(img_org, cv2.COLOR_BGR2HSV)

    ## Gen lower mask (0-5) and upper mask (175-180) of RED
    #these set works well
    mask1 = cv2.inRange(img_hsv, (0,50,50), (10,255,255))
    mask2 = cv2.inRange(img_hsv, (170,50,50), (180,255,255))

    # mask1 = cv2.inRange(img_hsv, (0,50,20), (5,255,255))
    # mask2 = cv2.inRange(img_hsv, (175,50,20), (180,255,255))

    ## Merge the mask and crop the red regions
    redmask = cv2.bitwise_or(mask1, mask2)
    #red_mask = cv2.bitwise_and(img_org, img_org, mask=mask)
    
    #red color mask 
    # lower_range = np.array([0,50,50]) #example value
    # upper_range = np.array([10,255,255]) #example value
    # red_mask = cv2.inRange(hsv_img, lower_range, upper_range)

    # yellow color mask
    lower_range = np.uint8([ 15,  38, 115])
    upper_range = np.uint8([ 35, 204, 255])
    yellow_mask = cv2.inRange(hsv_img, lower_range, upper_range)

    # white color mask
    lower_range = np.uint8([  0, 200,   0])
    upper_range = np.uint8([180, 255, 255])
    white_mask = cv2.inRange(hsv_img, lower_range, upper_range)

    combined_mask = white_mask | yellow_mask
    combined_mask = redmask | yellow_mask  
    #combined_mask = redmask
    #combined_mask = yellow_mask #only yellow mask  ->capable of filtering out most of the vehicle 
    masked_img = cv2.bitwise_and(img_org, img_org, mask=combined_mask)
    #masked_img = img_org  #no mask
    return masked_img

def perspective_transformation(img): 
    #IMAGE_H = 223
    IMAGE_H = 300
    IMAGE_W = 1280
    src = np.float32([[0, IMAGE_H], [1207, IMAGE_H], [0, 0], [IMAGE_W, 0]])
    #dst = np.float32([[543, IMAGE_H], [711, IMAGE_H], [0, 0], [IMAGE_W, 0]])
    dst = np.float32([[569, IMAGE_H], [711, IMAGE_H], [0, 0], [IMAGE_W, 0]])
    #dst = np.float32([[0, IMAGE_H], [1280, IMAGE_H], [0, 0], [1280, 0]])
    
    M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
    img = img[550:(550+IMAGE_H), 0:IMAGE_W] #crop the image
    warped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H)) # Image warping
    return warped_img

def get_vetices():
    new = np.array([[(0, 570), (496, 173), (596, 168), (1278, 400),(1280,960),(0,960)]], dtype=np.int32) #for testing vid01~05 except 02
    new = np.array([[(117, 955), (496, 473), (596, 468), (1278, 800),(1280,960)]], dtype=np.int32) # testing testingvid-04-truck.avi
    #new = np.array([[(150, 960), (451, 620), (855, 620), (1280, 810),(1280,960)]], dtype=np.int32)
    return new

def draw_dis_lines(img):
    img = np.zeros_like(img)
    #for speed estimation
    cv2.line(img,(50,614),(1260,614),(0,127,255),3)
    cv2.line(img,(50,684),(1260,684),(0,127,255),3)
    #cv2.line(img,(680,614),(680,684),(0,0,255),3)
    #for 
    cv2.line(img,(640,960),(575,545),(255,255,255),4)  #mid point 640 960
    cv2.line(img,(1000,960),(586,570),(255,255,0),3)  #shift 514 pixels
    cv2.line(img,(586,570),(500,570),(255,255,0),4)

    pol = np.array([[(224, 960), (500, 570), (586, 570),(1000, 960)]], dtype=np.int32)  

    cv2.line(img,(500,570),(224,960),(255,255,0),3)
    cv2.fillPoly(img,pol, (255,255,255))
    return img


def region_of_interest2(image, vertices):
    # defining a blank mask to start with
    mask = np.zeros_like(image)
    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(image.shape) > 2:
        channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    # returning the image only where mask pixels are non-zero
    masked_image = cv2.bitwise_and(image, mask)
    #masked_image = image  #without mask 
    return masked_image,mask

def filterout(image,vertices_polly):
    zero = np.zeros_like(image)
    cl = (255,255,255)
    cv2.fillPoly(zero, vertices_polly, cl)  #BGR
    filtered_image = cv2.bitwise_and(image, zero)
    #filtered_image = cv2.bitwise_and(image, mask)
    return filtered_image
    
#god
def filterout2(image,vertice,mask):
    zero = np.zeros_like(image)
    cl = (255,255,255)
    cv2.fillPoly(zero, vertice, cl)  #BGR
    #filtered_image = cv2.bitwise_and(image, zero)
    filtered_image = cv2.bitwise_and(image, zero)
    return filtered_image

def average_slope_intercept (image, lines):
    left = []
    right =[]
    lane = []
    for line in lines :
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2),1)
        #print("parameterssssssssssssss",parameters)
        try:
            slope, intercept = parameters
        except TypeError:
            slope, intercept = 0.001, 0 # It will minimize the error detecting the lane (putting 0, give you a math error)
        #slope = parameters[0]
        #intercept = parameters[1]
        
        if slope < -1 : #if slope < -1 
            left.append((slope,intercept))
        elif slope > 1:
            right.append((slope,intercept))

        if len(left) > 0 and len(right) < 1:
            lf = True
            print("only left")
            left_avg = np.average(left, axis=0)
            left_lane = make_coordinates(image, left_avg)
            right_lane = make_coordinates_append(image, left_avg)
            lane = np.array([left_lane,right_lane])
        else:
            left_lane = 0
            lf = False

        if len(right) > 0 and len(left) < 1:
            rt = True
            print("only right")
            right_avg = np.average(right, axis=0)
            right_lane = make_coordinates(image, right_avg)
            left_lane = make_coordinates_append(image,right_avg,left = True)
            lane = np.array([left_lane,right_lane])
        else:
            rt = False
            right_lane = 0

        if right and left :
            print("left and right")
            #left
            left_avg = np.average(left, axis=0)
            left_lane = make_coordinates(image, left_avg)
            #right
            right_avg = np.average(right, axis=0)
            right_lane = make_coordinates(image, right_avg)
            lane = np.array([left_lane,right_lane])    
    return lane ,lf ,rt


def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5)) #0.6 little bit too short to show the whole profile of vehicle to track
    #y2 = int(y1*(0.5))
    #y2 = int(y1*(1/5))  #stretch the line ->very long
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return np.array([x1, y1, x2, y2])

#append left or right lane when missing one  default:missing right lane
def make_coordinates_append(image, line_parameters,left = False):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5)) #0.6 little bit too short to show the whole profile of vehicle to track
    #y2 = int(y1*(0.5))
    #y2 = int(y1*(1/5))  #stretch the line ->very long
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    
    apd = np.array([x1+1100, y1, x2+140, y2])
    return np.array([x1+1100, y1, x2+140, y2]) if left == False else np.array([x1-1100, y1, x2-140, y2])

def canny(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    #reduce noise using gaussian filter 
    blur=cv2.GaussianBlur(gray, (5,5) , 0) #apply 5*5 kernal
    #Canny edge detection cv2.Canny(image , low_threshold , high_threshold), threshold: 
    canny = cv2.Canny(blur , 50 , 150)
    return canny
#for lines without averaging it    
def draw_lines1(image , lines):
    line_image = np.zeros_like(image)
    if lines is not None :
        #print("linessssssssss",lines)
        for line in lines :
            #print("#####",line)
            x1 , y1 , x2 , y2 = line.reshape(4)
            cv2.line(line_image, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 10)
    return line_image

def draw_lines2(image , lines):
    line_image = np.zeros_like(image)
    #print(lines,"lines in drawwwwwww")
    if lines is not None :
        try:
            for x1 , y1 , x2 , y2 in lines :
            #cv2.line(line_image, (int(x1),int(y1)), (int(x2),int(y2)), (255,0,0), 10)
                #print(x1,y1,x2,y2,"draw arg.............................")
                if x1 + y1 + x2 + y2 > 100000000 :
                    print("the value of lines are tooooo big ( in draw_lines2()  )")
                else :
                    cv2.line(line_image, (int(x1),int(y1)), (int(x2),int(y2)), (255,0,0), 10)
        except OverflowError:
            print("overflow!!")       
    return line_image

framenum = -1
def lane_detection(frame):
    vertices_polly = np.array([[(0, 0), (0, 0), (0, 0), (0, 0)]], dtype=np.int32)
    yellow_white = select_yellow_white(frame)
    cannyresult = canny(yellow_white)
    #cannyresult = canny(frame)
    #get the right vertice automatically
    vertice = get_vetices(frame)
    cropped_image , mask = region_of_interest2(cannyresult,vertice)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)  #minLineLength=40, maxLineGap=5
    #print("lines\n",lines)
    if lines is not None :
        lines = np.reshape(lines, [len(lines),4]) #lines will be None sometimes
        #avg_lines = average_slope_intercept(frame,lines)
        avg_lane, left , right = average_slope_intercept(frame,lines)
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
            #print(xr1,"777777777777777777777777")
            for x1 , y1 , x2 , y2 in avg_lane:
                xl1 , yl1 ,xl2 ,yl2 = x1 , y1 , x2 , y2

        elif right == True:
            for x1 , y1 , x2 , y2 in avg_lane:
                xr1 , yr1 ,xr2 ,yr2 = x1 , y1 , x2 , y2

        try:   
            #vertices_polly = np.array([[(xl1, yl1), (xl2, yl2), (xr2, yr2), (xr1, yr1)]], dtype=np.int32)
            vertices_polly = np.array([[(xl1, yl1), (xl2-5, yl2-100), (xr2, yr2-100), (xr1, yr1)]], dtype=np.int32) #extend trapezoid
        except (NameError,OverflowError):
            print("xl1 is not defined ->only one side of line works")
            
    else:
        print("default avg_lines(not detecting lanes)")
        avg_lane = np.array([[0 ,572 ,479 ,205],   #0 572 ; 479  205 ; 641 193 ; 1268 481

                             [1268 ,481 ,641 ,193]])
        vertices_polly = np.array([[(0, 0), (0, 0), (0, 0), (0, 0)]], dtype=np.int32)
        

    color_polly =  (0,255,0) #BGR
    line_image_not_avg = draw_lines1(frame, lines)
    
    line_image = draw_lines2(frame, avg_lane)
    god = filterout2(frame,vertice,mask)
    #print(vertices_polly)
    try:
        cv2.fillPoly(line_image, vertices_polly, color_polly)
        #cv2.fillPoly(zero_image, vertices_polly, 255)
        filtered = filterout(frame,vertices_polly)
        #god = filterout2(frame,mask)
    except NameError:
        #no_filtered = True
        print("vertices polly is not defined")
        filtered = cv2.addWeighted(frame, 1, line_image, 1, 1)
        filtered = cv2.addWeighted(filtered, 1, line_image_not_avg, 1, 1) #addWeighted function cant add two srcs
    #print(vertices_polly)
    return vertices_polly ,avg_lane

