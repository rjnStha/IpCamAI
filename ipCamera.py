import cv2

class ipCamera:
    def __init__(self, ip, username, password):
        self.ip= ip
        self.username = username
        self.password = password
        
    def videoLive(self):
        
        #Using RTSP to stream the ip camera
        video = cv2.VideoCapture("rtsp://" + self.username + 
                                 ":" + self.password + "@" + 
                                 self.ip + ":554/Streaming/Channels/1")
        
        while True:
            cv2.namedWindow("RTSP", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("RTSP", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            
            ret, frame = video.read()
            # Show the Image in the Window
            cv2.imshow("RTSP",frame)
            
            # Breaks the loop when 'q' key pressed, waitKey() returns code of key pressed
            if cv2.waitKey(1) == ord('q'):
                break
        
        video.release()
        cv2.destroyAllWindows()
    