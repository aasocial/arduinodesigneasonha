import cv2
import numpy as np
import time
import sys
import argparse
import RPi.GPIO as GPIO


COUNTER, FPS = 0, 0
START_TIME = time.time()
battery_counter = False

# Setting up the MG995 servo motor
GPIO.setmode(GPIO.BCM)
GPIO.setup(13, GPIO.out)

servo = GPIO.PWM(13, 50) # PWN frequency is 50Hz
servo.start(0)

def run(camera_id: int, width: int, height: int) -> None:
    global battery_counter
    """
    Continuously run the images acquired from the camera.

    Args:
        camera_id: camera id to be passed onto OpenCV
        width: width of the camera frame
        height: height of the camera frame
    """

    # Capture vide from the webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Visualisation variables
    row_size = 50  # pixels
    left_margin = 24
    text_color = (0,0,0)
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10
    shigan = 0

    def save_result() -> None:
        global FPS, COUNTER, START_TIME
        # Calculate the FPS
        if COUNTER % fps_avg_frame_count == 0:
            FPS = fps_avg_frame_count / (time.time() - START_TIME)
            START_TIME = time.time()

        COUNTER += 1

    while cap.isOpened():
        success, image = cap.read()
        global FPS, COUNTER, START_TIME
        # Calculate the FPS
        if COUNTER % fps_avg_frame_count == 0:
            FPS = fps_avg_frame_count / (time.time() - START_TIME)
            START_TIME = time.time()

        COUNTER += 1
        if not success:
            sys.exit(
                'ERROR: Unable to read from webcam. Please verify your webcam settings.'
            )

        image = cv2.flip(image, 1)

        fps_text = 'FPS = {:.1f}'.format(FPS)
        text_location = (left_margin, row_size)
        current_frame = image
        cv2.putText(current_frame, fps_text, text_location, cv2.FONT_HERSHEY_DUPLEX,
                    font_size, text_color, font_thickness, cv2.LINE_AA)

        blue = image[:,:,:1]
        green = image[:,:,1:2]
        red = image[:,:,2:]

        blue_mean = np.mean(blue)
        green_mean = np.mean(green)
        red_mean = np.mean(red)

        print("blue: " + str(blue_mean)+"\ngreen: " + str(green_mean) + "\nred: " + str(red_mean)+"\n" + str(shigan) + "\n")

        if blue_mean > 27 and green_mean > 27:
            battery_counter = True
            shigan += 1
        else:
            shigan = 0
            battery_counter = False

        angle_scale = 18
        if battery_counter == True and shigan > 22:
            servo.ChangeDutyCycle(85/angle_scale)

        if current_frame is not None:
            cv2.imshow("object_detection", current_frame)

        if cv2.waitKey(1) == 27:
            servo.ChangeDutyCycle(-85/angle_scale)
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--frameWidth',
        help='Width of frame to capture from camera.',
        required=False,
        type=int,
        default=1280
    )
    parser.add_argument(
        '--frameHeight',
        help='Height of frame to capture from camera.',
        required=False,
        type=int,
        default=720)
    args = parser.parse_args()

    run(0, args.frameWidth, args.frameHeight)

if __name__ == '__main__':
    main()
