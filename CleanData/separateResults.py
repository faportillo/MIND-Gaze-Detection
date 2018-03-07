"""
Assumes not all frames in a folder were annotated
this script moves all annotated frames into a separate folder 
use with restore .py to put remaning frames back into their original folder
"""

import os
import re 
from shutil import copy2

#TODO 
#add method for moving instead of copying
#create sys argv input to move instead of copy 


#path to folders with frames that have been annotated
LF = "F:/training set/lookingFinished"
NLF = "F:/training set/notLookingFinished"

#path to folders with all frames to be annotated
looking = "F:/training set/looking"
notLooking = "F:/training set/notLooking"

fileNames = os.listdir(looking)

columnNames = re.compile("(.*),(.*),(.*),(.*),(.*),(.*),(.*),(.*),(.*),(.*),(.*)") 
files = []

#path to csv that contains unformatted annotations 
with open("F:/training set/lookannotated.csv", "r") as look:
    lines = look.readline()

    for line in look:
        found = columnNames.match(line)
        video_file,x_init,y_inti,w,h,gaze_x,gaze_y,eye_x,eye_y,video_file1,video_file2 = found.groups()

        path = looking + "/" + video_file
        copy2(path, LF)

#path to csv that contains unformatted annotations 
with open("F:/training set/notlookingannotated.csv", "r") as look:
    lines = look.readline()

    for line in look:
        found = columnNames.match(line)
        video_file,x_init,y_inti,w,h,gaze_x,gaze_y,eye_x,eye_y,video_file1,video_file2 = found.groups()

        path = notLooking + "/" + video_file
        copy2(path, NLF)
        

            
print("finished")

