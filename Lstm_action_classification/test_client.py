import json
import requests

import cv2
import numpy as np
import json
import base64
import pickle
import math
import test

import sys

import cv2
from multiprocessing import Queue

import threading
import time



def draw_bodypose(canvas, candidate, subset):
    stickwidth = 4
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
               [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [
                  0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            # cv2.putText(canvas, str(index), (int(x), int(y)),cv2.FONT_HERSHEY_SIMPLEX,0.5, colors[n])
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)
    for i in range(17):
        for n in range(len(subset)):
            value1 = limbSeq[i][0]
            value2 = limbSeq[i][1]
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 1]
            X = candidate[index.astype(int), 0]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
            cv2.putText(canvas, str(value1)+","+str(value2), (int(mX),
                        int(mY)), cv2.FONT_HERSHEY_DUPLEX, 0.5, [0, 0, 0])
            polygon = cv2.ellipse2Poly((int(mX), int(mY)), (int(
                length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    return canvas


def main(video_location):
    jpeg_quality = 100
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    cam_label = "video data"
    # define a video capture object
    vid = cv2.VideoCapture(0)
    i = 0
    frame_skip = 5
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('output.mp4', fourcc, 10.0, (640, 480))
    while (vid.isOpened()):
        ret, image = vid.read()
        frame = cv2.resize(image.copy(), (640, 480))
        if ret != True:
            break
        result, buf = cv2.imencode('.jpg', frame, encode_params)
        bytes_data = buf.tobytes()
        img = np.frombuffer(bytes_data, dtype=np.dtype('uint8'))
        imdata = pickle.dumps(img)
        image_data = base64.b64encode(imdata).decode('ascii')
        payload = json.dumps({"image": image_data, "camera_id": cam_label})

        headers_face = {'Content-type': 'application/json',
                        'Accept': 'text/plain'}
        response = requests.post('http://0.0.0.0:8000',
                                 data=payload, headers=headers_face)
        res = response.json()
        # print("response ", res)

        # Use putText() method for
        # inserting text on video
        frame = draw_bodypose(
            frame, np.array(res["candidate"]), np.array(res["subset"]))
        x1 = 0
        y1 = 0
        x2 = 0
        y2 = 50
        color = (255, 0, 0)
        text_color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (w, h), _ = cv2.getTextSize(
            res["stride"], cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        # Prints the text.
        frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1+y2), color, -1)
        frame = cv2.putText(frame, res["stride"], (x1, y1+25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

        cv2.imshow("frame", frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        i += 1

    # After the loop release the cap object
    vid.release()
    out.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
    return ""


if __name__ == "__main__":
    video_location = sys.argv[1]
    # print(video_location)
    main(video_location)
