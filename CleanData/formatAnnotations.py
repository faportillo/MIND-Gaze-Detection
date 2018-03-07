"""
This script normalizes the dimension of the frame to 1x1 and adds 9 addtional locations where the baby might be looking 
"""
import re, csv
from numpy.random import randint

#path to csv with annotations that need to be formatted
newAnnotation = open("F:/training set/notLookingFinished/annotations.csv", "w", newline='')
csvWriter = csv.writer(newAnnotation)

#create new cvs with adjusted annotations
with open("F:/training set/notlookingannotated.csv", "r") as csvFile:
    csvReader = csv.reader(csvFile)
    row = next(csvReader)

    for row in csvReader:
        video0 = row[0]
        x_init = round(float(row[1]) / 720 , 3)
        y_init = round(float(row[2]) / 540, 3)
        w = round(float(row[3]) / 720, 3)
        h = round(float(row[4]) / 540, 3)
        gaze_x = float(row[5])
        gaze_y = float(row[6])
        eye_x = round(float(row[7]) / 720, 3)
        eye_y = round(float(row[8]) /540, 3)
        
        data = [video0, '1',str(x_init) ,str(y_init),str(w),str(h),str(round(gaze_x/720,3)), 
                str(round(gaze_y/540, 3)),str(eye_x),str(eye_y),row[0],row[0]]
        csvWriter.writerow(data)

        for i in range(2,11):
            try:
                x = min(max(randint(gaze_x - 10, gaze_x +10), 0), 720)
                y = min(max(randint(gaze_y - 10, gaze_y +10), 0), 540)
                x = str(round(x/720, 3))
                y = str(round(y/540, 3))
                
                data = [video0, str(i),str(x_init) ,str(y_init),str(w),str(h),x, 
                        y,str(eye_x),str(eye_y),row[0],row[0]]
                csvWriter.writerow(data)
            except(ValueError):
                print(gaze_x)

            

newAnnotation.close()