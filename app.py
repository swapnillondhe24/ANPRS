
import os
import time
from flask import Flask
from flask_restful import Resource, Api
from flask import request,Response
from werkzeug.utils import secure_filename

from flask_cors import CORS
import json

import base64
from PIL import Image
import io

import requests


from prediction import detect,detect_live


api = ''
app = Flask(__name__)
api = Api(app)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
cors = CORS(app)

app.config['CORS_HEADERS'] = 'Content-Type'
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

UPLOAD_DIR = "testing"
app.config['UPLOAD_FOLDER'] = UPLOAD_DIR


class detectLive(Resource):
   
    def post(self):
        try:
            # print("get data : ",request.get_data())

            # video_file = request.get_data()

            encoded_data = request.form['video']
            # print(encoded_data)
            # print(decoded_data)

            img = encoded_data.split(',')[1]
            decoded_data = base64.b64decode(img)

            image = Image.open(io.BytesIO(decoded_data))
            image.save("./testing/Live.jpeg", "JPEG")

            video_filename = "Live.jpeg"
            video_path = os.path.join(app.root_path, UPLOAD_DIR, video_filename)

            plates = detect(video_path)

            ret = json.dumps({"status" : "success","plates" : list(plates)})
            # print(ret)
            return Response(ret)


        
        except Exception as error:
            print(error)


class uploadVideo(Resource):
    
    def post(self):
        try:

            video_file = request.files['video']

            video_filename = secure_filename(video_file.filename)
            video_path = os.path.join(app.root_path, UPLOAD_DIR, video_filename)
            video_file.save(video_path)
            plates = detect(video_path)

            ret = json.dumps({"status" : "success","plates" : list(plates)})
            # print(ret)
            return Response(ret)
        
        except Exception as error:
            print(error)




api.add_resource(uploadVideo, '/uploadvideo')
api.add_resource(detectLive, '/detectlive')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5009)
    # app.run(debug=False)