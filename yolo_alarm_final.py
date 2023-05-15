import pyttsx3
import threading
import torch
import time
import cv2
from PIL import Image
import numpy as np
from audioplayer import AudioPlayer

# Load the Model
model = torch.hub.load('ultralytics/yolov5', 'custom', './weights/night_vision.pt')  # custom trained model
# Set the Confidence Threshold 
model.conf = 0.7



# Start VideoCapture for Webcam
video=cv2.VideoCapture(0)

# setting for AudioPlayer
wav_path = "siren_edit.wav"
# wav_path = "./siren.wav"
player_obj = AudioPlayer(wav_path)

# This funtion starts the Siren
def run_siren(player):
    player.play(block=True)
    player.close()

#Setting parameters for voice
# engine = pyttsx3.init() 
# voices = engine.getProperty('voices')
# engine.setProperty('voice', voices[1].id)
# engine.setProperty('rate', 150)

# This funtion plays the audio message
# def thread_voice_alert(engine):
    
#     engine.say("Animal Detected")
#     try:
#         engine.runAndWait()
#         engine.endLoop()
#         engine = None
#         engine.stop()
#     except:
#         pass

# Capture Frame and Run Object Detection Model
while True:
    # Frame reading
    check, frame = video.read()
    #frame = frame[:, :, [2,1,0]]
    # Convert an Image into Array
    frame = Image.fromarray(frame) 
    #frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
	
    # Test the Model on Webcam
    result = model(frame,size=640)
    # Save the Labels into a List
    labels = result.pandas().xyxy[0]['name'].values

    # Show the Detected Image
    cv2.imshow('YOLO', np.squeeze(result.render()))

    # If Any Object is Detected, Make sure to Alert
    if len(labels) != 0:
        # t = threading.Thread(target=thread_voice_alert, args=(engine,))
        t = threading.Thread(target=run_siren, args=(player_obj,))
        t.start()
    
    # Press 'q' to Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


#Clean up, Free memory
# engine.stop()
video.release()
cv2.destroyAllWindows