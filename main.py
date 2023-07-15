import os, sys
import cv2
import numpy as np
import math
import face_recognition
#from ipCamera import ipCamera

def faceConfidence(faceDistance, faceMatchThreshold = 0.6):
    range = (1.0 - faceMatchThreshold)
    linearVal = (1.0 - faceDistance) / (range * 2.0)
    
    if faceDistance > faceMatchThreshold:
        return str(round(linearVal *100,2)) + "%"
    else:
        value = (linearVal +((1.0 - linearVal) * math.pow((linearVal - 0.5)*2,0.2))) * 100
        return str(round(value, 2)) + "%"

class FaceRecognition:
    faceLocations = []
    faceEncodings = []
    faceNames = []
    knownFaceEncodings = []
    knownFaceNames = []
    #save computer power processing only other frame
    processCurrentFrame = True
    
    def __init__(self):
        self.encodeFaces()
        
    def encodeFaces(self):
        for image in os.listdir('faces'):
            faceImage = face_recognition.load_image_file(f'faces/{image}')
            faceEncoding = face_recognition.face_encodings(faceImage)[0]
            
            self.knownFaceEncodings.append(faceEncoding)
            self.knownFaceNames.append(image)
        
        print(self.knownFaceNames)
    
    def runRecognition(self):
        videoCapture = cv2.VideoCapture(0)
        
        if not videoCapture.isOpened():
            sys.exit('Video Source not Found!!')
            
        while True:
            ret, frame = videoCapture.read()
            
            # Only process every other frame of video to save process time
            if self.processCurrentFrame:
                # Resize video frame to 1/4th for faster recognition processing
                smallFrame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                
                # Convert BGR color(OpenCV) image to RGB(face_recognition) 
                rgbSmallFrame = cv2.cvtColor(smallFrame, cv2.COLOR_BGR2RGB)
                
                #Find all faces and face encodings in the current frame
                self.faceLocations = face_recognition.face_locations(rgbSmallFrame)
                self.faceEncodings = face_recognition.face_encodings(rgbSmallFrame, self.faceLocations)
                
                self.faceNames = []
                for faceEncoding in self.faceEncodings:
                    # match the known faces
                    matches = face_recognition.compare_faces(self.knownFaceEncodings, faceEncoding)
                    name = 'Unknown'
                    confidence = 'Unknown'
                    
                    # Calculate the shortest distance to face
                    faceDistances = face_recognition.face_distance(self.knownFaceEncodings, faceEncoding)
                    
                    bestMatchIndex = np.argmin(faceDistances)
                    if matches[bestMatchIndex]:
                        name = self.knownFaceNames[bestMatchIndex]
                        confidence = faceConfidence(faceDistances[bestMatchIndex])
                    
                    self.faceNames.append(f'{name}({confidence})')
            
            # to process every other frame and save process cost
            self.processCurrentFrame = not self.processCurrentFrame
            
            # Display the results
            for (top, right, bottom, left), name in zip(self.faceLocations, self.faceNames):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
               
                # Create the frame with the name
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
            
            # Display the resulting image
            cv2.imshow('Face Recognition', frame)
            
            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) == ord('q'):
                break
            
        # Release handle to the webcam
        videoCapture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    fr = FaceRecognition()
    fr.runRecognition()
    

#firstCam = ipCamera("192.168.0.3", "admin", "Bh1221bh")
#groundCam = ipCamera("192.168.0.4", "admin", "Bh1221bh")

#groundCam.videoLive()
#firstCam.videoLive()
