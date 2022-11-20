from flask import Flask, render_template, Response
from camera import VideoCamera
import os
import streamlit

app = Flask(__name__,template_folder='Templates')



@app.route('/')
def index():
    return render_template('index.html')


def gen(camera):
    while True:
        try:
            frame = camera.get_frame()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(frame) + b'\r\n\r\n')
        except Exception as e:
            pass


@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0',port=port, debug=False)