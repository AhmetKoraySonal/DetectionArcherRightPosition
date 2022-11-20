import cv2
import mediapipe as mp
import numpy as np
import time

class poseDetector():
    def __init__(self,mode=False,complexity=1,smootlandmarks=True,
                 ensegmentation=False,smtsegmentation=True,detectcon=0.5,trackcon=0.5):

        self.mode=mode
        self.complexity=complexity
        self.smoothlandmarks=smootlandmarks
        self.ensegmentation=ensegmentation
        self.smtsegmentation=smtsegmentation
        self.detectcon=detectcon
        self.trackcon=trackcon

        self.mpDraw=mp.solutions.drawing_utils
        self.mpPose=mp.solutions.pose
        self.pose=self.mpPose.Pose(self.mode, self.complexity,self.smoothlandmarks,
                                   self.ensegmentation,self.smtsegmentation,self.detectcon,
                                   self.trackcon)


    def findPose(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img,self.results.pose_landmarks)

        return img



    def GetPosition(self,img,name='Empty',draw=True):
        lmlist=[]
        ran=list(range(0,33))
        names=['nose','left_eye_inner','left_eye','left_eye_outer',
               'right_eye_inner','right_eye','right_eye_outer','left_ear',
              'right_ear','mouth_left','mouth_right','left_shoulder','right_shoulder',
              'left_elbow','right_elbow','left_wrist','right_wrist','left_pinky',
              'right_pinky','left_index','right_index','left_thumb','right_thumb','left_hip',
             'right_hip','left_knee','right_knee','left_ankle','right_ankle','left_heel',
              'right_heel','left_foot_index','right_foot_index']
        temp=dict(zip(names,ran))
        if name != "Empty":
            if self.results.pose_landmarks:
                for idd ,lm in enumerate(self.results.pose_landmarks.landmark):
                        h,w,c=img.shape
                        cx,cy=int(lm.x*w),int(lm.y*h)
                        lmlist.append([idd,cx,cy])
            return lmlist[temp[name]][1],lmlist[temp[name]][2]
        else:
            return [[],[]]

    def calculate_angle(self, image,joint_list):

        # Loop through hands
        for pose in self.results.pose_landmarks:
                # Loop through joint sets
            for joint in joint_list:
                a = np.array([pose.landmark[joint[0]].x, pose.landmark[joint[0]].y])  # First coord
                b = np.array([pose.landmark[joint[1]].x, pose.landmark[joint[1]].y])  # Second coord
                c = np.array([pose.landmark[joint[2]].x, pose.landmark[joint[2]].y])  # Third coord

                radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
                angle = np.abs(radians * 180.0 / np.pi)

                if angle > 180.0:
                    angle = 360 - angle
                print(angle)
                cv2.putText(image, str(round(angle, 2)), tuple(np.multiply(b, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        return image


#def main():
#    cap = cv2.VideoCapture(0)
#    ptime = 0
#    detector=poseDetector()
#    while True:
#        success, image = cap.read()
#        image=detector.findPose(image,draw=False)
#        #lmlist=detector.GetPosition(image,draw=False)
#        x,y=detector.GetPosition(image,name='nose',draw=False)

        #if len(lmlist)!=0:
        #    print(lmlist)


#        cTime = time.time()  # time class diğer time function
#        fps = 1 / (cTime - ptime)
#        ptime = cTime

#        cv2.putText(image, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
#        cv2.imshow('Image', image)
#        cv2.waitKey(1)



#if __name__ =="__main__":
#    main()
