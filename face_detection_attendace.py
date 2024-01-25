import pandas as pd
import cv2
import requests
import urllib.request
import numpy as np
import os
from datetime import datetime
import face_recognition
import database as db

path = 'image_folder'
url='http://192.168.8.101/cam-hi.jpg'

if 'Attendance.csv' in os.listdir(os.path.join(os.getcwd(),'attendance')):
    print("there iss..")
    os.remove("Attendance.csv")
else:
    df = pd.DataFrame(list())
    df.to_csv("Attendance.csv")
    
users_id = db.db_get_userid_list()
for user_id in users_id:
    db.db_get_photo(path, user_id)

images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    with open("Attendance.csv", 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name},{dtString}')


encodeListKnown = findEncodings(images)
print('Encoding Complete')

#cap = cv2.VideoCapture(0)

while True:
    #success, img = cap.read()
    img_resp=urllib.request.urlopen(url)
    imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
    img=cv2.imdecode(imgnp,-1)
# img = captureScreen()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            # --------------------------------------------------------------------------------------
            p_id = classNames[matchIndex].upper()
            names = db.db_get_names(p_id)
            p_status = db.db_get_status(p_id)
            # user_id=333666&name=Ivan&surname=Mischenko&status_enter=out
            data_str = f"user_id={p_id}&name={names[0]}&surname={names[1]}&status_enter={p_status}"
            print(data_str)
            r = requests.post('http://192.168.8.101:80/data', data=data_str, headers={'Content-Type': 'application/x-www-form-urlencoded'})
            print(r.text)
            # --------------------------------------------------------------------------------------
            name = f'{names[0]} {names[1]}'
            print(name) #commented
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow('Webcam', img)
    key = cv2.waitKey(5)
    if key == ord('q'):
        for user_id in users_id:
            ph_dir = f'{path}/{user_id}.jpg'
            os.remove(ph_dir)
        break

cv2.destroyAllWindows()
cv2.imread