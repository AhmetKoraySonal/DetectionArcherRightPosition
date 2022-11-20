import cv2
import mediapipe as mp
import numpy as np
import time
from google.protobuf.json_format import MessageToDict

class HandDetector():

    def __init__(self,mode=False,maxHands=2,detectionCon=0.5,trackCon=0.5,model_complexity=1):
        self.mode=mode
        self.maxHands=maxHands
        self.detectionCon=detectionCon
        self.trackCon=trackCon
        self.model_complexity=model_complexity

        self.mpHands=mp.solutions.hands
        self.hands=self.mpHands.Hands(self.mode,self.maxHands,self.model_complexity,self.detectionCon,self.trackCon)
        self.mpDraw=mp.solutions.drawing_utils

    def findHands(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms)
        return img


    def FindPosition(self,img,left_or_right,HandNo=0,name='Empty',draw=True):
        lmList=[]
        ran=list(range(0,21))
        names=['WRIST','THUMB_CMC','THUMB_MCP','THUMB_IP','THUMB_TIP','INDEX_FINGER_MCP',
               'INDEX_FINGER_PIP','INDEX_FINGER_DIP','INDEX_FINGER_TIP','MIDDLE_FINGER_MCP',
               'MIDDLE_FINGER_PIP','MIDDLE_FINGER_DIP','MIDDLE_FINGER_TIP','RING_FINGER_MCP',
               'RING_FINGER_PIP','RING_FINGER_DIP','RING_FINGER_TIP','PINKY_MCP','PINKY_PIP',
               'PINKY_DIP','PINKY_TIP']
        temp=dict(zip(names,ran))
        if name!='Empty':
            if self.results.multi_hand_landmarks:  #detection olup olmadıgı kontrol edilir
                if len(self.results.multi_handedness) != 2:
                    for i in self.results.multi_handedness:
                        label = MessageToDict(i)[
                        'classification'][0]['label']
                        if label == 'Right' and left_or_right=='Left':
                            myHand=self.results.multi_hand_landmarks[HandNo]
                            for id, lm in enumerate(myHand.landmark):  # her bir noktanın id sı ve pozisyonu elde edilir
                                h, w, c = img.shape
                                cx, cy = int(lm.x * w), int(lm.y * h)
                                lmList.append([id,cx,cy])
                        if label=='Left' and left_or_right=='Right':
                            myHand = self.results.multi_hand_landmarks[HandNo]
                            for id, lm in enumerate(myHand.landmark):  # her bir noktanın id sı ve pozisyonu elde edilir
                                h, w, c = img.shape
                                cx, cy = int(lm.x * w), int(lm.y * h)
                                lmList.append([id, cx, cy])

            return lmList[temp[name]][1],lmList[temp[name]][2]
        else:
            return [[],[]]

    def FindAngle(self,image,joint_list):
            # Loop through hands
        for hand in self.results.multi_hand_landmarks:
                # Loop through joint sets
            for joint in joint_list:
                a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y])  # First coord
                b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y])  # Second coord
                c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y])  # Third coord

                radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
                angle = np.abs(radians * 180.0 / np.pi)

                if angle > 180.0:
                    angle = 360 - angle

                cv2.putText(image, str(round(angle, 2)), tuple(np.multiply(b, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        return image


    def FindDistance(self,x,y,shape_x,shape_y):
        distance = (((x[0] - y[0]) / shape_x) ** 2 + ((x[1] - y[1]) / shape_y) ** 2) ** 0.5

        return distance
#def main():
#   pTime = 0
#    cTime = 0
    #    cap=cv2.VideoCapture(0)
    #    hand_detector=HandDetector()

        #   while True:
        #     success, img = cap.read()
        #    img=hand_detector.findHands(img,draw=False)
            #lmlist=hand_detector.FindPosition(img,name='THUMP_TIP',draw=False)
        #print(lmlist[1],lmlist[2])
            # try:
            #    x,y=hand_detector.FindPosition(img,name='WRIST',draw=False)
        #    print(x,y)
            # except:
        #    pass

        # cTime = time.time()  # time class diğer time function
        # fps = 1 / (cTime - pTime)
        # pTime = cTime

        #  cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        #  cv2.imshow('Image', img)
#  cv2.waitKey(1)



#if __name__ =="__main__":
#    main()
