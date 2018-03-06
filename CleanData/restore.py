import os
annotatedFramesDir = "F:\\annotated_frames"
Frames12Dir = "F:\\Frames12"

FrameList = os.listdir(Frames12Dir)
annotatedFramesList = os.listdir(annotatedFramesDir)

Frames12Dir.split
for frame in FrameList:
    fileName = frame.split(" ")[0]
    if fileName in annotatedFramesList:
        source = os.path.join(Frames12Dir,frame)
        target = os.path.join(annotatedFramesDir,fileName,frame )
        os.rename(source, target )