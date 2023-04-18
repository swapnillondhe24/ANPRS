
import time
from flask import Flask
from flask_restful import Resource, Api
from flask import request,Response
from flask_cors import CORS
import json

import requests


api = ''
app = Flask(__name__)
api = Api(app)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
cors = CORS(app)

app.config['CORS_HEADERS'] = 'Content-Type'
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


class startCamera(Resource):
    def startCamera(self):
        time.sleep(5)
        return json.dumps({"status" : "success"})

    
    def post(self):
        try:
            request_json = request.get_json()
            query = request_json["query"]

            return Response(self.searchRecpie(query))
            # return Response(request_json)
        except Exception as error:
            print(error)


class uploadVideo(Resource):
    def uploadVideo(self):
        time.sleep(5)
        return json.dumps({"status" : "success"})

    
    def post(self):
        try:
            request_json = request.get_json()
            query = request_json["query"]

            return Response(self.searchRecpie(query))
            # return Response(request_json)
        except Exception as error:
            print(error)




api.add_resource(uploadVideo, '/uploadvideo')
api.add_resource(startCamera, '/startcamera')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5006)
    # app.run(debug=False)