import os,re
from math import floor

#frame rate
rate = 29.97

#"sub_id","visit","video_file","code","actor","code_time","state_end"
columns = re.compile("(.*),(.*),(.*),(.*),(.*),(.*),(.*)")
codesFile = "F:\\coding_data\\look_face_codes.csv"

#create a dictionary with all videos and timestamps of looking children
files = {}
with open(codesFile, 'r') as cF:
    line = cF.readline()
    for line in cF:
        found = columns.match(line)
        sub_id,visit,video_file,code,actor,code_time,state_end = found.groups()
        video_file = video_file.strip('".mpg')
        code = code.strip('"')
        actor = actor.strip('"')
        if actor == 'Parent':
            continue
        elif video_file in files:
            files[video_file].append((float(code_time), float(state_end)))
        else:
            files[video_file] = []
            files[video_file].append((float(code_time), float(state_end)))

#grab frames from each file and put them into a folder 
# one for looking and one for not looking
looking = "F:\\looking"
notLooking = "F:\\notLooking"
frameDir = os.listdir("F:\\annotated_frames")
if not os.path.exists(notLooking):
    print("{} doesn't exist".format(notLooking))

#loop through all video folders
for video, times in files.items():
    counter = 0
    if video in frameDir:
        currentFolder = "F:\\annotated_frames\\" + video
        frameList = os.listdir(currentFolder)
        lastStop = times[0][1]

        #loop through all looking times in a video folder
        #only grab 12 frames; 2 each cylce
        for start, stop in times: 
            if counter == 6:
                break

            #grab look frame and put in look folder
            frameIndex = int(floor((start + stop)/2) * rate)
            try:
                lookingFrame = frameList[frameIndex]
                lookSource = currentFolder + "\\" + lookingFrame
                lookTarget = looking + "\\" + lookingFrame 
                os.rename(lookSource, lookTarget)
            except (IndexError):
                print("looking {} {}".format(video,frameIndex))
                continue
            
            #grab a frame between looking times
            if lastStop != stop:
                frameIndex = int(floor((lastStop + start)/2 * rate)) + 30
                if frameIndex < len(frameList) and frameIndex < int(start*rate):
                    try:
                        notLookingFrame = frameList[frameIndex]
                        notLookingSource = currentFolder + "\\" + notLookingFrame
                        notLookingTarget = notLooking + "\\" + notLookingFrame
                        if not os.path.exists(notLookingSource):
                            print("{} doesn't exist".format(notLookingSource))
                            continue
                        os.rename(notLookingSource, notLookingTarget)
                        lastStop = stop
                    except(IndexError):
                        print("not looking {} {}".format(video, frameIndex))
                        continue
                counter += 1

        #move last frame not looking    
        frameIndex = int(stop * rate + 10)
        if frameIndex < len(frameList):
            try:
                notLookingFrame = frameList[frameIndex]
                notLookingSource = currentFolder + "\\" + notLookingFrame
                notLookingTarget = notLooking + "\\" + notLookingFrame  
                if not os.path.exists(notLookingSource):
                            print("{} doesn't exist".format(notLookingSource))
                            continue
                os.rename(notLookingSource, notLookingTarget)    
            except(IndexError):
                print("not looking {} {}".format(video, frameIndex))
                continue


print("finished")
    


        
        