
# import os
from matplotlib import pyplot as plt
import cv2
import numpy as np
import easyocr
import pymongo

# Replace the placeholders with your connection string and database name
connection_string = "mongodb+srv://Admin:Admin@anprs.csdqtst.mongodb.net/?retryWrites=true&w=majority"
db_name = "ANPRS"
# Create a MongoClient object with the connection string
client = pymongo.MongoClient(connection_string)

    # Access the database using the client and database name
db = client[db_name]

    # Access a collection and query for a document by LP_number
collection = db["LicensePlates"]








#predictions

INPUT_WIDTH =  640
INPUT_HEIGHT = 640


# LOAD YOLO MODEL
net = cv2.dnn.readNetFromONNX('./Model/weights/best.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)




# initialize the EasyOCR reader
reader = easyocr.Reader(['en'])

def extract_text(image,bbox):
    x,y,w,h = bbox
    roi = image[y:y+h, x:x+w]
    
    if 0 in roi.shape:
        return 'no number'
    
    else:
        # extract text using EasyOCR
        result = reader.readtext(roi)
        text = ' '.join([res[1] for res in result])
        text = text.strip()
        
        return text


def get_detections(img,net):
    # 1.CONVERT IMAGE TO YOLO FORMAT
    image = img.copy()
    row, col, d = image.shape

    max_rc = max(row,col)
    input_image = np.zeros((max_rc,max_rc,3),dtype=np.uint8)
    input_image[0:row,0:col] = image

    # 2. GET PREDICTION FROM YOLO MODEL
    blob = cv2.dnn.blobFromImage(input_image,1/255,(INPUT_WIDTH,INPUT_HEIGHT),swapRB=True,crop=False)
    net.setInput(blob)
    preds = net.forward()
    detections = preds[0]
    
    return input_image, detections

def non_maximum_supression(input_image,detections):
    
    # 3. FILTER DETECTIONS BASED ON CONFIDENCE AND PROBABILIY SCORE
    
    # center x, center y, w , h, conf, proba
    boxes = []
    confidences = []

    image_w, image_h = input_image.shape[:2]
    x_factor = image_w/INPUT_WIDTH
    y_factor = image_h/INPUT_HEIGHT

    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4] # confidence of detecting license plate
        if confidence > 0.4:
            class_score = row[5] # probability score of license plate
            if class_score > 0.25:
                cx, cy , w, h = row[0:4]

                left = int((cx - 0.5*w)*x_factor)
                top = int((cy-0.5*h)*y_factor)
                width = int(w*x_factor)
                height = int(h*y_factor)
                box = np.array([left,top,width,height])

                confidences.append(confidence)
                boxes.append(box)

    # 4.1 CLEAN
    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()
    
    # 4.2 NMS
    index = cv2.dnn.NMSBoxes(boxes_np,confidences_np,0.25,0.45)
    
    return boxes_np, confidences_np, index

def drawings(image,boxes_np,confidences_np,index):
    # 5. Drawings
    texts = []
    for ind in index:
        x,y,w,h =  boxes_np[ind]
        bb_conf = confidences_np[ind]
        conf_text = 'plate: {:.0f}%'.format(bb_conf*100)
        license_text = extract_text(image,boxes_np[ind])
        texts.append(license_text)


        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),2)
        cv2.rectangle(image,(x,y-30),(x+w,y),(255,0,255),-1)
        cv2.rectangle(image,(x,y+h),(x+w,y+h+25),(0,0,0),-1)


        cv2.putText(image,conf_text,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),1)
        cv2.putText(image,license_text,(x,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

    return image, texts


# predictions flow with return result
def yolo_predictions(img,net):
    # step-1: detections
    input_image, detections = get_detections(img,net)
    # step-2: NMS
    boxes_np, confidences_np, index = non_maximum_supression(input_image, detections)
    # step-3: Drawings
    result_img, texts = drawings(img,boxes_np,confidences_np,index)
    return result_img, texts




def detect(file):
    import re
    state_list = ("AN","AP","AR","AS","BH","BR","CH","CG","DD","DL","GA","GJ","HR","HP","JK","JH","KA","KL","LA","LD","MP","MH","MN","ML","MZ","NL","OD","PY","PB","RJ","SK","TN","TS","TR","UP","UK","WB")
    cap = cv2.VideoCapture(file)
    text_results = []
    while cap.isOpened():
        ret, frame = cap.read()

        if ret == False:
            print('Unable to read video or it ended')
            break

        results, texts = yolo_predictions(frame,net)
        # print(texts)
        # print(texts)
        texts = [text.upper() for text in texts]
        texts = ''.join(texts)

        texts_filtered = re.sub(r'\W+', '', texts)
        # print(texts_filtered)
        if texts_filtered:
          # texts = texts.split(" ")
          # print(texts_filtered)
          if texts_filtered[:2] in state_list:
            if re.match(r'[A-Z\d]{3,4}[A-Z\d]{2}\d{4}', texts_filtered):
                    print(texts_filtered)
                    text_results.append(texts_filtered)
                    name = find_document(texts_filtered)
                    if name:
                        return {"status" : "success","data" : name}

                    # {"status" : "success","plates" : list(plates)}
        
        for i in range(2):
                ret = cap.grab()
    
    print(text_results)
    return_frame = {
            "Name" : "Unknown",
            "Plate": set(text_results)
            }
    # {"status" : "success","plates" : list(plates)}
    return {"status" : "success","data" : return_frame}

def write_to_file(file_path, lines):
    with open(file_path, 'w') as f:
        for line in lines:
            f.write(line + '\n')

def detect_live(file):
    import re
    state_list = ("AN","AP","AR","AS","BH","BR","CH","CG","DD","DL","GA","GJ","HR","HP","JK","JH","KA","KL","LA","LD","MP","MH","MN","ML","MZ","NL","OD","PY","PB","RJ","SK","TN","TS","TR","UP","UK","WB")
    cap = cv2.VideoCapture(file)
    text_results = []
    while cap.isOpened():
        ret, frame = cap.read()

        if ret == False:
            print('Unable to read video or it ended')
            break

        results, texts = yolo_predictions(frame,net)
        # print(texts)
        # print(texts)
        texts = [text.upper() for text in texts]
        texts = ''.join(texts)

        texts_filtered = re.sub(r'\W+', '', texts)
        # print(texts_filtered)
        if texts_filtered:
          # texts = texts.split(" ")
          # print(texts_filtered)
          if texts_filtered[:2] in state_list:
            if re.match(r'[A-Z\d]{3,4}[A-Z\d]{2}\d{4}', texts_filtered):
                    print(texts_filtered)
                    text_results.append(texts_filtered)
                    name = find_document(texts_filtered)
                    if name:
                        return {"status" : "success","data" : name}

                    # {"status" : "success","plates" : list(plates)}
        
        for i in range(2):
                ret = cap.grab()
    
    print(text_results)
    return_frame = [{
            "Name" : "",
            "Plate": set(text_results)}]
    # {"status" : "success","plates" : list(plates)}
    return {"status" : "success","data" : return_frame}       
    
def find_document(lp_number):
    

    # lp_number = '"'+lp_number+'"'
    query = {"LP_number": lp_number}
    
    documents = collection.find(query)

    # If the document exists, return Name and LP_number
    ret = []
    for document in documents:
        ret.append({
            "Name" : document["Name"],
            "Plate": document["LP_number"]
        })

    # If the document does not exist, return None
    return ret


if __name__ == "__main__":
    detect_live()
