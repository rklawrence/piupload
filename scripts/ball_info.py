#!/usr/bin/env python3

import cv2
import imutils
import rospy
from ball_detection.msg import ballInfo
from sensor_msgs.msg import Image
import numpy as np


class BallInfo:
    def __init__(self, color, x, y, radius):
        self.color = color
        self.x = x
        self.y = y
        self.radius = radius

class BallDetector:
    def __init__(self):
        rospy.init_node('ball_info')
        self.publisher = rospy.Publisher("/ballInfo", ballInfo, queue_size=10)
        rospy.Subscriber("/usb_cam/image_raw", Image, self.handle_image_data)

    @staticmethod
    def get_camera(camera_number=0):
        """_summary_
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
        Parameters
        ----------
        camera_number: int
            The index of the webcam that you want to view, defaults
            to the 0 which is the default camera of the machine

        Returns
        ----------
        camera: cv2.VideoCapture
            The camera that is opened with cv2
        """
        camera = cv2.VideoCapture(camera_number)
        return camera
        # while(True):
        #     _, frame = camera.read()
        #     cv2.imshow("frame", frame)

        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break

        # camera.release()
        # cv2.destroyAllWindows()

    @staticmethod
    def detect_black_and_white(img):
        """_summary_

        Args:
            input_img (_type_): _description_
            params (_type_): _description_

        Returns:
            _type_: _description_
        """
        img = cv2.equalizeHist(img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_yellow = (0, 0, 250)
        upper_yellow = (1, 1, 255)
        lower_green = (0, 0, 170)
        upper_green = (1, 1, 200)
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        img = cv2.bitwise_and(img, img, mask=mask)
        img = mask
        return img

    @staticmethod
    def detect_color(img):
        """_summary_

        Args:
            img (_type_): _description_

        Returns:
            yellow_mask
                A black and white image that has isolated the pixels that should contain a yellow ball
            green_mask
                A black and white image that has isolated the pixels that should contain a green ball
            purple_mask
                A black and white image that has isolated the pixels that should contain a purple ball
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img = hsv

        low_yellow = (90, 0, 200)
        high_yellow = (110, 200, 255)
        low_green = (30, 150, 50)
        high_green = (55, 255, 200)
        low_purple = (130, 70, 0)
        high_purple = (160, 200, 200)
        low_blue = (15, 230, 150)
        high_blue = (30, 255, 255)

        yellow_mask = cv2.inRange(hsv, low_yellow, high_yellow)
        green_mask = cv2.inRange(hsv, low_green, high_green)
        purple_mask = cv2.inRange(hsv, low_purple, high_purple)
        blue_mask = cv2.inRange(hsv, low_blue, high_blue)

        return yellow_mask, green_mask, purple_mask, blue_mask

    @staticmethod
    def get_contours(img, mask, color=None):
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        center = None

        if len(contours) > 0:
            circle = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(circle)
            moment = cv2.moments(circle)
            center = (int(moment["m10"] / moment["m00"]), int(moment["m01"] / moment["m00"]))

            if radius > 10:
                cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(img, center, 5, (0, 0, 255), -1)
            ball_info = BallInfo(color, x, y, radius)
        else:
            ball_info = None
        return img, ball_info

    def handle_image_data(self, data):
        image_data = data.data
        image = np.frombuffer(image_data, dtype=np.uint8).reshape(data.height, data.width, -1)
        colors = self.detect_color(image)
        color_names = ["yellow", "green", "purple", "blue"]
        for index, color in enumerate(colors):
            _, ball_info = self.get_contours(image.copy(), color, color_names[index])
            if ball_info is None:
                continue
            color = ball_info.color
            x = ball_info.x
            y = ball_info.y
            radius = ball_info.radius
            msg = ballInfo()
            msg.color = color
            msg.x = int(x)
            msg.y = int(y)
            msg.radius = int(radius)
            self.publisher.publish(msg)

    def main_loop(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            rate.sleep()


if __name__ == '__main__':
    node = BallDetector()
    node.main_loop()
