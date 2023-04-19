
import os
import time
from flask import Flask
from flask_restful import Resource, Api
from flask import request,Response
from werkzeug.utils import secure_filename

from flask_cors import CORS
import json

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


class startCamera(Resource):
   
    def post(self):
        try:
            return Response(detect_live(0), mimetype='text/event-stream')
        except Exception as error:
            print(error)


class stopCamera(Resource):
   
    def post(self):
        try:
            return Response(detect_live(1), mimetype='text/event-stream')
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
api.add_resource(startCamera, '/startcamera')
api.add_resource(stopCamera, '/stopcamera')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5009)
    # app.run(debug=False)