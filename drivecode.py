import cv2
import pyautogui
from time import time  
from math import hypot
import mediapipe as mp
import matplotlib.pyplot as plt


mp_pose = mp.solutions.pose
pose_image = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=1)
pose_video = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

def detectPose(image, pose, draw=False, display=False):  
    output_image = image.copy()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(imageRGB)

    if results.pose_landmarks and draw:
        mp_drawing.draw_landmarks(
            image=output_image,
            landmark_list=results.pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=3, circle_radius=3),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(49, 125, 237), thickness=2, circle_radius=2) 
        )

    if display:
        plt.figure(figsize=[22, 22])
        plt.subplot(121); plt.imshow(image[:, :, ::-1]); plt.title("Original Image"); plt.axis("off")
        plt.subplot(122); plt.imshow(output_image[:, :, ::-1]); plt.title("Output Image"); plt.axis('off')
        plt.show()  
    else:
        return output_image, results

def checkHandsJoined(image, results, draw=False, display=False):  
    height, width, _ = image.shape
    output_image = image.copy()

    left_wrist_landmark = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * width,
                          results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * height)
    
    right_wrist_landmark = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * width,
                           results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * height)
    
    euclidean_distance = int(hypot(left_wrist_landmark[0] - right_wrist_landmark[0],
                                left_wrist_landmark[1] - right_wrist_landmark[1]))
    
    if euclidean_distance < 130:  
        hand_status = 'Hands Joined'  
        color = (0, 255, 0)
    else:  
        hand_status = 'Hands Not Joined'  
        color = (0, 0, 255)

    if draw:
        cv2.putText(output_image, hand_status, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 3)
        cv2.putText(output_image, f'Distance: {euclidean_distance}', (10, 70),
                   cv2.FONT_HERSHEY_PLAIN, 2, color, 3)
    if display: 
        plt.imshow(output_image[:, :, ::-1])
        plt.title("Hands Joined Detection")
        plt.axis('off')
        plt.show()
    else:
        return output_image, hand_status


camera_video = cv2.VideoCapture(0)  
camera_video.set(3, 1280)  
camera_video.set(4, 960)  

cv2.namedWindow('Hands Joined', cv2.WINDOW_NORMAL) 

while camera_video.isOpened():
    ok, frame = camera_video.read()
    if not ok:
        continue

    frame = cv2.flip(frame, 1)  
    frame, results = detectPose(frame, pose_video, draw=True)

    if results.pose_landmarks:
        frame, _ = checkHandsJoined(frame, results, draw=True)

    cv2.imshow('Hands Joined', frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:  
        break

camera_video.release()
cv2.destroyAllWindows()

def checkLeftRight(image, results, draw=False, display=False):
    horizontal_position = None
    height, width, _ = image.shape
    output_image = image.copy()

    # Get shoulder landmarks (fixed left/right mixup)
    left_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * width)
    right_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * width)

    # Determine horizontal position (fixed logic)
    mid_x = width // 2
    if right_x <= mid_x and left_x <= mid_x:
        horizontal_position = 'Left'
    elif right_x >= mid_x and left_x >= mid_x:
        horizontal_position = 'Right'
    else:
        horizontal_position = 'Center'

    if draw:
        cv2.putText(output_image, f'Position: {horizontal_position}', 
                    (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
        cv2.line(output_image, (mid_x, 0), (mid_x, height), (255, 255, 255), 2)

    if display:
        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:, :, ::-1])
        plt.title("Horizontal Position Detection")
        plt.axis('off')
        plt.show()
    else:
        return output_image, horizontal_position

def checkJumpCrouch(image, results, MID_Y=250, draw=False, display=False):
    height, width, _ = image.shape
    output_image = image.copy()

    # Get shoulder landmarks
    left_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * height)
    right_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * height)
    actual_mid_y = (left_y + right_y) // 2

    # Define dynamic bounds (better than fixed values)
    lower_bound = MID_Y - 50  # Jump threshold
    upper_bound = MID_Y + 50  # Crouch threshold

    # Determine posture
    if actual_mid_y < lower_bound:
        posture = 'Jumping'
    elif actual_mid_y > upper_bound:
        posture = 'Crouching'
    else:
        posture = 'Standing'

    if draw:
        cv2.putText(output_image, f'Posture: {posture}', 
                    (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
        cv2.line(output_image, (0, MID_Y), (width, MID_Y), (255, 255, 255), 2)

    if display:
        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:, :, ::-1])
        plt.title("Vertical Movement Detection")
        plt.axis('off')
        plt.show()
    else:
        return output_image, posture

# Main execution
camera_video = cv2.VideoCapture(0)  # Use 0 for default camera
camera_video.set(3, 1280)  # Width
camera_video.set(4, 960)   # Height

cv2.namedWindow('Movement Detection', cv2.WINDOW_NORMAL)

while camera_video.isOpened():
    ok, frame = camera_video.read()
    if not ok:
        continue

    frame = cv2.flip(frame, 1)  # Mirror effect
    frame, results = detectPose(frame, pose_video, draw=True)

    if results.pose_landmarks:
        # Check both horizontal and vertical movements
        frame, _ = checkLeftRight(frame, results, draw=True)
        frame, _ = checkJumpCrouch(frame, results, MID_Y=frame.shape[0]//2, draw=True)

    cv2.imshow('Movement Detection', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

camera_video.release()
cv2.destroyAllWindows()

pyautogui.press(keys='up')

pyautogui.press(keys=["up","up","up","up","down"])

pyautogui.press(key="down",presses=4)

pyautogui.keyDown(key='shift')

pyautogui.press(keys="enter",presses=2)

pyautogui.keyUp(key="shift")

print("hello")
print('happy learning ')

pyautogui.keyDown(key='ctrl')

pyautogui.press(keys='tab')

pyautogui.keyUp(key='ctrl')

pyautogui.click(button="right")

pyautogui.click(x=1300,y=800,button='right')

camera_video = cv2.VideoCapture(2)
camera_video.set(3,1280)
camera_video.set(4,960)

cv2.namedWindow('Subway Sufers with pose detection ', cv2.WINDOW_NORMAL)

time=0
game_started=False
x_pos_index=1
y_pos_index=1
MID_Y=None
counter=0
num_of_frames=20
while camera_video.isOpened():
    ok, frame=camera_video.read()

    if not ok:
        continue

    frame=cv2.flip(frame,1)
    frame_height, frame_width,_= frame.shape
    frame, results=detectPose(frame,pose_video,draw= game_started)

    if results.pose_landmarks:
        if game_started:
            frame, horizontal_position=  checkLeftRight(frame,results,draw=True)

            if(horizontal_position== "Left" and x_pos_index!=0)or(horizontal_position== "Center" and x_pos_index==2):
                pyautogui.press('left')

                x_pos_index-=1

            elif(horizontal_position== "Right" and x_pos_index!=2)or(horizontal_position== "Center" and x_pos_index==2):  
                pyautogui.press('Right')

                x_pos_index+=1

            if checkHandsJoined(frame,results)[1]=='Hands Joined':
                pyautogui.press('space')

            else:
                cv2.putText(frame,"JOIN BOTH HANDS TO START THE GAME",(5,frame_height-10),cv2.FONT_HERSHEY_PLAIN,
                            2,(0,255,0),3)
                counter+=1
                if counter == num_of_frames:
                    game_started=True

                    left_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame_height)
                    right_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame_height)
                    MID_Y = abs(left_y + right_y) // 2
                else:
                    counter=0
        
        if MID_Y :
            frame , posture = checkJumpCrouch(frame,results,MID_Y, draw=True)

            if posture == "Jumping" and y_pos_index==1:
                pyautogui.press("up")

                y_pos_index-=1

            elif posture=="Standing" and y_pos_index !=1:

                y_pos_index=1
    else:

        counter=0

        time2=time()

        if(time2-time1)>0:

            frame_per_second = 1.0/ (time2 - time1)
            cv2.putText(frame,"FPS:{}".format(int(frame_per_second)),(10,30),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),3)

        time1= time2

        cv2.imshow('subway sufers with pose detection ', frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  
            break

camera_video.release()
cv2.destroyAllWindows()