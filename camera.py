import math

import cv2
import HandTrackingModule as ht
import BodyTrackingModule as bt
import numpy as np


class VideoCamera(object):

    epsilon = 8
    ##Sag el ile atılırsa ok açı
    R_SolKoltuk_altı = list(range(int(88.49708221227748 - epsilon), int(88.49708221227748 + epsilon)))
    R_SagKoltuk_altı = list(range(int(127.42614987292467 - epsilon), int(127.42614987292467 + epsilon)))
    R_SolDirsekIc = list(range(int(176.9059419410829 - epsilon), int(176.9059419410829 + epsilon)))
    R_SagDirsekIc = list(range(int(21.043743569671477 - epsilon), int(21.043743569671477 + epsilon)))
    ##Sol el ile atılırsa  ok açı
    S_SagKoltuk_altı = list(range(int(88.49708221227748 - epsilon), int(88.49708221227748 + epsilon)))
    S_SolKoltuk_altı = list(range(int(127.42614987292467 - epsilon), int(127.42614987292467 + epsilon)))
    S_SagDirsekIc = list(range(int(176.9059419410829 - epsilon), int(176.9059419410829 + epsilon)))
    S_SolDirsekIc = list(range(int(21.043743569671477 - epsilon), int(21.043743569671477 + epsilon)))

    joint_list = [[4, 3, 2], [8, 7, 6], [12, 11, 10], [16, 15, 14], [20, 19, 18]]
    joint_list1 = [[12, 14, 16], [11, 13, 15], [23, 11, 13], [24, 12, 14]]

    hand_detector = ht.HandDetector()
    body_detector = bt.poseDetector()
    a_l, a_r, b_l, b_r, c_l, c_r, d_l, d_r = (0, 0, 0, 0, 0, 0, 0, 0)

    El_secim = 'Left'

    def __init__(self, interval=20,Secim='Left'):
        self.video = cv2.VideoCapture(0)
        self.El_secim = Secim
        self.interval = interval
        self.wrist_finger=False
        self.sagkoltukaltı=False
        self.solkoltukaltı = False
    def __del__(self):
        self.video.releast()

    def SagdanAtis(self, imgg):
        a_r = 0
        b_r = 0
        c_r = 0
        d_r = 0
        if int(self.angle1) in self.R_SagDirsekIc:
            print('Sag dirseginizin ic acisi dogru')
            a_r = 1
        else:
            if int(self.angle1) >= self.R_SagDirsekIc[-1]:
                print('Sag dirseginizin ic acisini %d derece indirin' % (abs(int(self.angle1 - self.R_SagDirsekIc[-1]))))
                imgg = self.AddArrow('static\\LeftRedArrow.jpeg', imgg, [self.leftelbow_x, self.leftelbow_y])

            else:
                print('Sag dirseginizin ic acisini %d kaldırın' % (abs(int(self.angle1 - self.R_SagDirsekIc[0]))))
                imgg = self.AddArrow('static\\RightGreenArrow.jpeg', imgg, [self.rightelbow_x, self.rightelbow_y])

        if int(self.angle2) in self.R_SolDirsekIc:
            print('Sol dirseginizin iç acisi dogru')
            b_r = 1
        else:
            if int(self.angle2) >= self.R_SolDirsekIc[-1]:
                print('Sol dirseginizin ic acisini %d indirin' % (abs(int(self.angle2 -self.R_SolDirsekIc[-1]))))
                imgg = self.AddArrow('static\\LeftRedArrow.jpeg', imgg, [self.leftelbow_x, self.leftelbow_y])

            else:
                print('Sol dirseginizin ic acisini %d kaldırın' % (abs(int(self.angle2 - self.R_SolDirsekIc[0]))))
                imgg = self.AddArrow('static\\RightGreenArrow.jpeg', imgg, [self.leftelbow_x, self.leftelbow_y])

        if int(self.angle3) in self.R_SolKoltuk_altı:
            print('Sol koltukaltınızın acısı dogru')
            c_r = 1
        else:
            if int(self.angle3) >= self.R_SolKoltuk_altı[-1]:
                print('Sol koltukaltınızı  %d derece indirin' % (abs(int(self.angle3 - self.R_SolKoltuk_altı[-1]))))
                imgg = self.AddArrow('static\\DownRedArrow.jpeg', imgg, [self.leftshoulder_x, self.leftshoulder_y])

            else:
                print('Sol koltukaltınızı %d derece kaldırın' % (abs(int(self.angle3 - self.R_SolKoltuk_altı[0]))))
                imgg = self.AddArrow('static\\UpGreenArrow.jpeg', imgg, [self.leftshoulder_x, self.leftshoulder_y])

        if int(self.angle4) in self.R_SagKoltuk_altı:
            print('Sag koltukaltınızın acisi dogru')
            d_r = 1
        else:
            if int(self.angle4) >= self.R_SagKoltuk_altı[-1]:
                print('Sag koltukaltınızı %d derece indirin' % (abs(int(self.angle4 - self.R_SagKoltuk_altı[-1]))))
                imgg = self.AddArrow('static\\DownRedArrow.jpeg', imgg, [self.rightshoulder_x, self.rightshoulder_y])

            else:
                print('Sag koltukaltinizi %d derece kaldırın' % (abs(int(self.angle4 - self.R_SagKoltuk_altı[0]))))
                imgg = self.AddArrow('static\\UpGreenArrow.jpeg', imgg, [self.rightshoulder_x, self.rightshoulder_y])


        if (a_r + b_r + c_r + d_r) == 4:
            imgg = self.Addtick('static\\greentick.jpeg', imgg)

        return imgg

    def SoldanAtis(self, imgg):
        a_l = 0
        b_l = 0
        c_l = 0
        d_l = 0
        if int(self.angle1) in self.S_SagDirsekIc:
            print('Sag dirsek açınız dogru')
            a_l = 1
        else:
            if int(self.angle1) >= self.S_SagDirsekIc[-1]:
                print('Sag dirseginizin ic acisini %d derece azaltın' % (abs(int(self.angle1 - self.S_SagDirsekIc[-1]))))
                imgg = self.AddArrow('static\\LeftRedArrow.jpeg', imgg, [self.rightelbow_x, self.rightelbow_y])

            else:
                print('Sag dirseginizin ic acisini %d derece arttırın' % (abs(int(self.angle1 - self.S_SagDirsekIc[0]))))
                imgg = self.AddArrow('static\\RightGreenArrow.jpeg', imgg, [self.rightelbow_x, self.rightelbow_y])

        if int(self.angle2) in self.S_SolDirsekIc:
            print('Sol dirsek açınız dogru')
            b_l = 1
        else:
            if int(self.angle2) >= self.S_SolDirsekIc[-1]:
                print('Sol dirseginizin ic acisini %d derece azaltın' % (abs(int(self.angle2 - self.S_SolDirsekIc[-1]))))
                imgg = self.AddArrow('static\\LeftRedArrow.jpeg', imgg, [self.leftelbow_x, self.leftelbow_y])

            else:
                print('Sol dirseginizin ic acisini %d derece arttırın' % (abs(int(self.angle2 - self.S_SolDirsekIc[0]))))
                imgg = self.AddArrow('static\\RightGreenArrow.jpeg', imgg, [self.leftelbow_x, self.leftelbow_y])


        if int(self.angle3) in self.S_SolKoltuk_altı:
            print('Sol koltukaltı açısı dogru')
            c_l = 1
        else:
            if int(self.angle3) >= self.S_SolKoltuk_altı[-1]:
                print('Sol koltukaltınızı %d derece indirin' % (abs(int(self.angle3 - self.S_SolKoltuk_altı[-1]))))
                imgg = self.AddArrow('static\\DownRedArrow.jpeg', imgg, [self.leftshoulder_x, self.leftshoulder_y])

            else:
                print('Sol koltukaltınızı %d derece kaldırın' % (abs(int(self.angle3 - self.S_SolKoltuk_altı[0]))))
                imgg = self.AddArrow('static\\UpGreenArrow.jpeg', imgg, [self.leftshoulder_x, self.leftshoulder_y])

        if int(self.angle4) in self.S_SagKoltuk_altı:
            print('Sag koltuk altinizin acisi dogru')
            d_l = 1
        else:
            if int(self.angle4) >= self.S_SagKoltuk_altı[-1]:
                print('Sag koltuk altınızı %d derece indirin' % (abs(int(self.angle4 - self.S_SagKoltuk_altı[-1]))))
                imgg = self.AddArrow('static\\DownRedArrow.jpeg', imgg, [self.rightshoulder_x, self.rightshoulder_y])

            else:
                print('Sag koltuk altınızı  %d derece kaldırın' % (abs(int(self.angle4 - self.S_SagKoltuk_altı[0]))))
                imgg = self.AddArrow('static\\UpGreenArrow.jpeg', imgg, [self.rightshoulder_x, self.rightshoulder_y])


        if (a_l + b_l + c_l + d_l) == 4:
            imgg = self.Addtick('static\\greentick.jpeg', imgg)

        return imgg
    def calltick(self, path,size=(50,50)):
        tick = cv2.imread(path)
        tick = cv2.resize(tick, size)
        return tick

    def Addtick(self, path, image):
        tick = self.calltick(path)
        img2gray = cv2.cvtColor(tick, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
        roi = image[-50 - 10:-10, -50 - 10:-10]
        roi[np.where(mask)] = 0
        roi += tick
        return image

    def FindAngle(self, a, b, c):
        a = np.array([a[0], a[1]])  # First coord
        b = np.array([b[0], b[1]])  # Second coord
        c = np.array([c[0], c[1]])  # Third coord

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        # cv2.putText(image, str(round(angle, 2)), tuple(np.multiply(b, [640, 480]).astype(int)),
        # cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        return angle

    def AddArrow(self, path,image, coor, size=(30, 30)):
        arroww = self.calltick(path, size)
        coor[0] = int(coor[0])
        coor[1] = int(coor[1])
        img2gray = cv2.cvtColor(arroww, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)

        if (int((coor[1] - size[1] / 2) >= 0)) and ((int(coor[0] - size[0] / 2)) >= 0) and (int(coor[1] + size[1] / 2) <= image.shape[0]) and (int(coor[0] + size[0] / 2) <= image.shape[1]):
            roi = image[coor[1] - int(size[1] / 2):coor[1] + int(size[1] / 2),
                  coor[0] - int(size[0] / 2):coor[0] + int(size[0] / 2)]
        roi[np.where(mask)] = 0
        roi += arroww
        return image

    def list_contains(self,List1, List2):
        check = False
        for m in List1:
            for n in List2:
                if m == n:
                    check = True
                    return check
        return check

    def select_area(self,underarm_angle, intervall):
        intervall = list(range(int(underarm_angle - intervall), int(underarm_angle + intervall)))
        return intervall

    def distance_wrist_and_finger(self,aa, bb,cc,dd):
        # a lefteyeinner b lefteyeouter c ile d uzaklık tespit
        distance1=math.sqrt((aa[1] - bb[1]) ** 2 + (bb[0]-aa[0]) ** 2)
        distance2=math.sqrt((cc[1] - dd[1]) ** 2 + (cc[0]-dd[0]) ** 2)
        ratio=float(abs(distance1))/float(abs(distance2))
        return ratio

    def get_frame(self):

        ret, frame = self.video.read()
        h, w, c = frame.shape
        try:
            frame = self.hand_detector.findHands(frame, draw=False)
            ##Hand##
            frame = self.hand_detector.FindAngle(frame, self.joint_list)
            self.middle_x,self.middle_y = self.hand_detector.FindPosition(frame, self.El_secim, name='MIDDLE_FINGER_TIP',
                                                                 draw='False')
            self.ring_x,self.ring_y= self.hand_detector.FindPosition(frame, self.El_secim, name='RING_FINGER_TIP', draw='False')
            self.wrist_x,self.wrist_y= self.hand_detector.FindPosition(frame, self.El_secim, name='WRIST',draw='False')
            self.indexfingermcp_x,self.indexfingermcp_y=self.hand_detector.FindPosition(frame, self.El_secim, name='INDEX_FINGER_MCP',draw='False')
            self.pinkymcp_x, self.pinkymcp_y = self.hand_detector.FindPosition(frame, self.El_secim,name='PINKY_MCP',draw='False')
            middle_wrist_ratio = self.distance_wrist_and_finger([self.indexfingermcp_x,self.indexfingermcp_y],
                                                   [self.pinkymcp_x, self.pinkymcp_y],
                                                   [self.middle_x, self.middle_y], [self.wrist_x, self.wrist_y])

            ring_wrist_ratio=self.distance_wrist_and_finger([self.indexfingermcp_x,self.indexfingermcp_y],
                                                   [self.pinkymcp_x, self.pinkymcp_y],
                                                   [self.ring_x, self.ring_y], [self.wrist_x, self.wrist_y])

            if (middle_wrist_ratio)>0.6 and (ring_wrist_ratio)>0.6:
                self.wrist_finger=True
            else:
                self.wrist_finger=False


        except:
            pass

        try:
            frame = self.body_detector.findPose(frame, draw=False)


            ##Body##
            # (12,14,16)
            self.rightshoulder_x, self.rightshoulder_y = self.body_detector.GetPosition(frame, 'right_shoulder', draw=False)
            self.rightelbow_x, self.rightelbow_y = self.body_detector.GetPosition(frame, 'right_elbow', draw=False)
            self.rightwrist_x, self.rightwrist_y = self.body_detector.GetPosition(frame, 'right_wrist', draw=False)

            # (11,13,15)
            self.leftshoulder_x, self.leftshoulder_y = self.body_detector.GetPosition(frame, 'left_shoulder', draw=False)
            self.leftelbow_x, self.leftelbow_y = self.body_detector.GetPosition(frame, 'left_elbow', draw=False)
            self.leftwrist_x, self.leftwrist_y = self.body_detector.GetPosition(frame, 'left_wrist', draw=False)
            # (23,11,13)
            self.lefthip_x, self.lefthip_y = self.body_detector.GetPosition(frame, 'left_hip', draw=False)
            # (24,12,14)
            self.righthip_x,self. righthip_y = self.body_detector.GetPosition(frame, 'right_hip', draw=False)

            self.angle1 = self.FindAngle([self.rightshoulder_x, self.rightshoulder_y], [self.rightelbow_x,self. rightelbow_y],
                                    [self.rightwrist_x, self.rightwrist_y])
            self.angle2 = self.FindAngle([self.leftshoulder_x, self.leftshoulder_y], [self.leftelbow_x, self.leftelbow_y],
                                    [self.leftwrist_x, self.leftwrist_y])
            self.angle3 = self.FindAngle([self.lefthip_x, self.lefthip_y], [self.leftshoulder_x, self.leftshoulder_y],
                                    [self.leftelbow_x, self.leftelbow_y])
            self.angle4 = self.FindAngle([self.righthip_x, self.righthip_y], [self.rightshoulder_x, self.rightshoulder_y],
                                    [self.rightelbow_x, self.rightelbow_y])

            if self.El_secim=='Left':  # sol ile atılıyor
                frame=self.SoldanAtis(frame)
            else:  # sag el ile atılıyor
                frame=self.SagdanAtis(frame)


        except:
            pass
        ret, jpeg = cv2.imencode('.jpeg', frame)
        return jpeg.tobytes()
