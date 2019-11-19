from imutils.video import FPS
import numpy as np
import argparse
import imutils
import cv2
# import v4l2 to get faster frame rate with webcam, it seems openCV can't set it
# use openCV v4l2 backend, reinstallation needed

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=False,
                help="path to input video file")
args = vars(ap.parse_args())

# open a pointer to the video stream and start the FPS timer
# stream = cv2.VideoCapture(args["video"])
stream = cv2.VideoCapture(0, apiPreference=cv2.CAP_V4L2)
fps = FPS().start()
c = 0

stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
stream.set(cv2.CAP_PROP_FPS, 60)
# stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# loop over frames from the video file stream
while True:
    # grab the frame from the threaded video file stream
    (grabbed, frame) = stream.read()

    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break
    if c > 300:
        break
    # resize the frame and convert it to grayscale (while still
    # retaining 3 channels)
    frame = imutils.resize(frame, width=450)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = np.dstack([frame, frame, frame])

    # display a piece of text to the frame (so we can benchmark
    # fairly against the fast method)
    cv2.putText(frame, "Slow Method", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # show the frame and update the FPS counter
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)
    fps.update()
    c += 1


# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
stream.release()
cv2.destroyAllWindows()
