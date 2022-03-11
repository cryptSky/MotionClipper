# import the necessary packages
import argparse
import datetime
import imutils
import time
import numpy as np
import cv2
import copy
import xml.etree.ElementTree as ET
from urllib.parse import unquote
import uuid
from tqdm import tqdm, trange
from PyQt5 import QtCore
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QMutex

ticks_in_frame = 10584000000

class MotionClipper(QObject):
    progressValueUpdated = pyqtSignal(int)
    progressTextUpdated = pyqtSignal(str)
    finishedUpdated = pyqtSignal(int)
    sgnFinished = pyqtSignal()

    def __init__(self, filename):
        super(MotionClipper, self).__init__()
        self._mutex = QMutex()
        self._running = True
        
        self.project_xml_file = filename
        self.tree = ET.parse(self.project_xml_file)
   
    def getProgressPercent(self, frame_number, num_frames):
        return (frame_number / num_frames) * 100
        
    @pyqtSlot()
    def stop(self):
        self._mutex.lock()
        self._running = False
        self._mutex.unlock()
        
    @pyqtSlot()
    def start(self):
        self._mutex.lock()
        self._running = True
        self._mutex.unlock()
    
    def running(self):
        try:
            self._mutex.lock()
            return self._running
        finally:
            self._mutex.unlock()
            
    def seekPosition(self, video, currentState, min_area, alpha, threshold, width, startFrameNumber, endFrameNumber):
    
        print("Current state:", currentState)
        print("Start Frame:", startFrameNumber)
        print("End Frame:", endFrameNumber)
        if endFrameNumber - startFrameNumber == 1 or endFrameNumber - startFrameNumber == -1:
            return endFrameNumber
    
        if currentState == "Unoccupied":
            rg = range(startFrameNumber, endFrameNumber)
        else:
            rg = range(endFrameNumber, startFrameNumber, -1)            
        
        for frame_number in rg: 
            print("Frame:", frame_number)
            position = frame_number
            
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            (grabbed, frame) = video.read()
            
            # resize the frame, convert it to grayscale, and blur it
            frame = imutils.resize(frame, width=width)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            if frame_number == rg[0]:
                firstFrame = gray
                continue
            
            # compute the absolute difference between the current frame and
            # first frame
            frameDelta = cv2.absdiff(firstFrame, gray)
            thresh = cv2.threshold(frameDelta, threshold[0], threshold[1], cv2.THRESH_BINARY)[1]
            firstFrame = (alpha*gray).astype(np.uint8) + ((1-alpha)*firstFrame).astype(np.uint8)
            
            # dilate the thresholded image to fill in holes, then find contours
            # on thresholded image
            thresh = cv2.dilate(thresh, None, iterations=2)
            (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
                
            if len(cnts) == 0:
                continue
        
            # loop over the contours
            for c in cnts:
                # if the contour is too small, ignore it
                if cv2.contourArea(c) < min_area:
                    continue
                else:
                    print("Position: ", position)
                    return position
            
                
        return position
        
    def postprocess_cuts(self, movements, stills, minNonMotionFrames, nonMotionBeforeStart, nonMotionAfter, minFramesToKeep):
        #if minNonMotionFrames <= nonMotionBeforeStart:
        #    return movements, stills

        intervals_bf  = []

        intervals_bf.append(list(stills[0]))

        for i in range(len(movements) -1):
            intervals_bf.append( [movements[i][0], movements[i][1]] )          
            intervals_bf.append( [stills[i+1][0] - nonMotionBeforeStart, stills[i+1][1]] )

        intervals_bf.append( list(movements[len(movements) - 1]) )

        if nonMotionBeforeStart > 0:            

            #print("Intervals_bf: ")
            #print(intervals_bf)

            i = 1
            merged = []
            merged.append(intervals_bf[0])

            while i < len(intervals_bf):
                len_merged = len(merged)
                last = merged[len_merged-1]
                current = intervals_bf[i]

                if last[1] == current[0]:
                    merged.append(current)
                    i += 1
                    continue

                j = 1
                while current[0] < intervals_bf[i-j][1] and i - j >= 0:
                    if j == 2:
                        k = 0    
                    last = merged.pop()           
                    j += 1
                
                if (i-j-1) % 2 == 1:
                    last[1] = current[0]
                    if last[0] == last[1]:
                        merged[len(merged)-1][1] = current[1]
                    else:
                        merged.append(last)
                        merged.append(current)                
                else:
                    last[1] = current[1]
                    merged.append(last)
            
                i += 1

        else:
            merged = intervals_bf

        #print("Merged: ")
        #print(merged)

        intervals_af  = []        
        #end = if len(merged) % 2 == 0 end = len(merged) - 1 else end = len(merged)

        if nonMotionAfter > 0:
            i = 0
            while i < len(merged):
                len_merged = merged[i][1] - merged[i][0]
                total_len = 0

                if i + 1 == len(merged):
                    intervals_af.append( merged[i] )
                    break

                j = 1
                while i + j < len(merged) - 1 and total_len + merged[i+j][1] - merged[i+j][0] <= nonMotionAfter:
                    total_len += merged[i+j][1] - merged[i+j][0]           
                    j += 1            

                if j % 2 == 0:
                    intervals_af.append( [merged[i][0], merged[i+j][1]] )
                else:
                    steps = nonMotionAfter - total_len

                    intervals_af.append( [merged[i][0], merged[i+j][0] + steps] ) 
                    intervals_af.append( [merged[i+j][0] + steps, merged[i+j][1]] )


                i += j + 1
        else:
            intervals_af = merged

        print("intervals_af: ")
        print(intervals_af)

        intervals = intervals_af

        movements_kept = [] 
        stills_kept = []   
        stills_kept.append(intervals[0])
        start = intervals[0]

        prev_motion = False

        i = 1
        while i < len(intervals):
            length = intervals[i][1]-intervals[i][0]
            end = intervals[i][1]

            len_movements = len(movements_kept)
            len_stills = len(stills_kept)

            if length >= minFramesToKeep:
                start = intervals[i][0]
                if i % 2 == 1: 
                    if len_movements > 0 and movements_kept[len_movements-1][1] == start:
                        movements_kept[len_movements-1][1] = end
                    else:
                        movements_kept.append( [start, end] )

                    prev_motion = True                 
                else:
                    if len_stills > 0 and stills_kept[len_stills-1][1] == start:
                        stills_kept[len_stills-1][1] = end
                    else:
                        stills_kept.append( [start, end] ) 

                    prev_motion = False
            else:
                if prev_motion:                   
                    movements_kept[len_movements-1][1] = end
                else:
                    stills_kept[len_stills-1][1] = end                          
            
            i += 1
        
        return movements_kept, stills_kept
    
    def detect_movement(self, filename, show_detection=False, min_area=500, alpha=0.2, threshold=(32, 255), 
    width=1000, minMotionFrames=30, minNonMotionFrames=30, nonMotionBeforeStart=0, nonMotionAfter=0, minFramesToKeep=0):
        print("Parameters passed:")
        print(show_detection, min_area, alpha, threshold, width, minMotionFrames, minNonMotionFrames, nonMotionBeforeStart, nonMotionAfter, minFramesToKeep)
                
        video = cv2.VideoCapture(filename)
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # initialize the first frame in the video stream
        firstFrame = None
        
        current_state = "Unoccupied"
        
        stills = []
        movements = []
        
        start = 0
        end = 0

        frame_number = 0
        frame_step = minNonMotionFrames
        
        frame = 0
        
        while frame is not None:
            self.progressValueUpdated.emit(self.getProgressPercent(frame_number, num_frames))
            
            if not self._running:
                video.release()
                if show_detection:
                    cv2.destroyWindow("Thresh")
                    cv2.destroyWindow("Frame Delta")
                    cv2.destroyWindow("Motion Detection Feed")
                self.sgnFinished.emit()
                return None, None, None, True
            
            # grab the current frame and initialize the occupied/unoccupied
            # text
            
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            (grabbed, frame) = video.read()

            if frame is None:
                break
            
            # resize the frame, convert it to grayscale, and blur it
            if width > 0:
                frame = imutils.resize(frame, width=width)
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
            # if the first frame is None, initialize it
            if firstFrame is None:
                firstFrame = gray
                continue
        
            # compute the absolute difference between the current frame and
            # first frame
            frameDelta = cv2.absdiff(firstFrame, gray)
            thresh = cv2.threshold(frameDelta, threshold[0], threshold[1], cv2.THRESH_BINARY)[1]
            firstFrame = (alpha*gray).astype(np.uint8) + ((1-alpha)*firstFrame).astype(np.uint8)
        
            # dilate the thresholded image to fill in holes, then find contours
            # on thresholded image
            thresh = cv2.dilate(thresh, None, iterations=2)
            (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
        
            if current_state == "Occupied" and len(cnts) == 0:
                
                #startFrameNumber = frame_number - 2*minMotionFrames + 1
                #endFrameNumber = frame_number - minMotionFrames
                
                end = frame_number #self.seekPosition(video, current_state, min_area, alpha, threshold, width, startFrameNumber, endFrameNumber)
                movements.append((start, end))
        
                current_state = "Unoccupied"
                start = end
                #frame_number = end
                frame_step = minNonMotionFrames
                        
            # loop over the contours
            for c in cnts:
                # if the contour is too small, ignore it
                if cv2.contourArea(c) < min_area:
                    continue
        
                if current_state == "Unoccupied":
                
                    #startFrameNumber = frame_number - minNonMotionFrames + 1
                    #endFrameNumber = frame_number
                
                    end = frame_number #self.seekPosition(video, current_state, min_area, alpha, threshold, width, startFrameNumber, endFrameNumber - 1)
                    
                    #end = frame_number
                    if end > 0:
                        stills.append((start, end))
        
                    start = end
                    #frame_number = end
                    current_state = "Occupied"
                    frame_step = minMotionFrames
        
                # compute the bounding box for the contour, draw it on the frame,
                # and update the text
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
            if show_detection:
                # draw the text and timestamp on the frame
                cv2.putText(frame, "State: {}".format(current_state), (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            
                # show the frame and record if the user presses a key
                cv2.imshow("Thresh", thresh)
                cv2.imshow("Frame Delta", frameDelta)
                cv2.imshow("Motion Detection Feed", frame)
                key = cv2.waitKey(1) & 0xFF
            
                # if the `q` key is pressed, break from the lop
                if key == ord("q"):
                    break
            
                           
            frame_number += frame_step
            #pbar.update(frame_number)
        
        while frame is None:
            frame_number -= 1
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            (grabbed, frame) = video.read()
            
        if current_state == "Occupied":
            #print((start, frame_number - 1))
            movements.append((start, frame_number))
        else:
            #print((start, frame_number - 1))
            stills.append((start, frame_number))
        
        # cleanup the camera and close any open windows
        video.release()
        
        if show_detection:
            cv2.destroyWindow("Thresh")
            cv2.destroyWindow("Frame Delta")
            cv2.destroyWindow("Motion Detection Feed")

        print("With movements: ")
        print(movements)
        
        print("Without movements: ")
        print(stills)

        print(show_detection, min_area, alpha, threshold, width, minMotionFrames, minNonMotionFrames, nonMotionBeforeStart, nonMotionAfter, minFramesToKeep)
            
        movements, stills = self.postprocess_cuts(movements, stills, minNonMotionFrames, nonMotionBeforeStart, nonMotionAfter, minFramesToKeep)
        
        print("With movements: ")
        print(movements)
        
        print("Without movements: ")
        print(stills)
        
        return frame_number, movements, stills, False
    
    
    def remove_tracks(self, root):
        if root.getchildren() is not None:
            for child in root.getchildren():
                if child.tag == 'track':
                    root.remove(child)
                else:
                    self.remove_tracks(child)
                    
    @pyqtSlot()
    def process(self, show_detection=False, min_area=500, alpha=0.2, threshold=(32, 255), width=1000, minMotionFrames=30, minNonMotionFrames=30, nonMotionBeforeStart=0, nonMotionAfter=0, minFramesToKeep=0):
        root = self.tree.getroot()

        print(show_detection, min_area, alpha, threshold, width, minMotionFrames, minNonMotionFrames, nonMotionBeforeStart, nonMotionAfter, minFramesToKeep)
        
        sequence = root.findall("./sequence")[0]
        clipitems = len(root.findall("./sequence/media/video/track/clipitem"))
        #files = len(root.findall("./sequence/media/video/track/clipitem/file"))
        
        new_sequence = copy.deepcopy(sequence)
        new_sequence.attrib["id"] = new_sequence.attrib["id"] + "-motion-clipped"
        guid = str(uuid.uuid4())
        new_sequence.find("uuid").text = guid
        
        new_sequence.find("name").text = new_sequence.find("name").text + " - Motion clipped"
        self.remove_tracks(new_sequence)    
    
        tracks = sequence.findall("./media/video/track")
        for track in tracks:
        
            movements_track = copy.deepcopy(track)
            clips = movements_track.findall("./clipitem")
            
            
            for clip in clips:        
                movements_track.remove(clip)            
            movements_track.attrib["MZ.TrackName"] = "GH4"
                    
            stills_track = copy.deepcopy(track)
            clips = stills_track.findall("./clipitem")
            for clip in clips:
                stills_track.remove(clip)
            stills_track.attrib["MZ.TrackName"] = "GH5"
            
            clips = track.findall("./clipitem")
            start_frame = 0
            
            file_data_saved = False
            for index, clip in enumerate(clips):
                if not file_data_saved:
                    file = copy.deepcopy(clip.find("./file"))
                    empty_file = copy.deepcopy(file)
                    for child in list(empty_file):
                        empty_file.remove(child)
                        
                    file_data_saved = True
            
                #if len(clip.find("./file").getchildren()) > 0:
                #    nc = copy.deepcopy(clip)
                #    file = copy.deepcopy(nc.find("./file"))
                #    file.attrib["id"] = "file-"+str(files+1)
                #    empty_file = copy.deepcopy(file)
                #    for child in list(empty_file):
                #        empty_file.remove(child)
                #    empty_file.attrib["id"] = "file-"+str(files+1)
                #    #empty_file = nc.find("file")
                #    files += 1
                    
                    #print(ET.tostring(file, encoding='utf8').decode('utf8'))
                    #print(ET.tostring(empty_file, encoding='utf8').decode('utf8'))
    
                
                file_path_url = clip.find("./file/pathurl").text
                file_path = unquote(file_path_url)
                file_path = file_path[file_path.index("/", 8) + 1:]
                file_name = file_path[file_path.rindex("/")+1:]
                msg = "Processing file " + file_name + " ... (" + str(index+1) +"/" + str(len(clips)) + ")"
                print(msg)
                
                self.progressTextUpdated.emit(msg)
                total_frames, movements, stills, stopped = self.detect_movement(file_path, show_detection, min_area, alpha, threshold, width, minMotionFrames, minNonMotionFrames, nonMotionBeforeStart, nonMotionAfter, minFramesToKeep)
                
                if stopped:
                    return
            
                file_set = False
                for id, movement in enumerate(movements):                
                    new_clip_item = copy.deepcopy(clip)
                    if not file_set:
                        new_clip_item.remove(new_clip_item.find("file"))
                        new_clip_item.append(file)
                        file_set = True
                    else:
                        new_clip_item.remove(new_clip_item.find("file"))
                        new_clip_item.append(empty_file)
                            
                    new_clip_item.find("start").text = str(start_frame+movement[0])
                    new_clip_item.find("end").text = str(start_frame+movement[1])
                    new_clip_item.find("in").text = str(movement[0])
                    new_clip_item.find("out").text = str(movement[1])
                    new_clip_item.find("pproTicksIn").text = str(ticks_in_frame*movement[0])
                    new_clip_item.find("pproTicksOut").text = str(ticks_in_frame*movement[1])
                    
                    new_clip_item.find("./labels/label2").text = "Violet"
                    new_clip_item.attrib["id"] = "clipitem-"+str(clipitems+id+1)
                    
                    movements_track.append(new_clip_item)
                    
                for index, still_frame in enumerate(stills):
                    new_clip_item = copy.deepcopy(clip)
                    new_clip_item.remove(new_clip_item.find("file"))
                    if not file_set:
                        if new_clip_item.find("file") is not None:
                            new_clip_item.remove(new_clip_item.find("file"))
                        new_clip_item.append(file)
                        file_set = True
                    else:
                        if new_clip_item.find("file") is not None:
                            new_clip_item.remove(new_clip_item.find("file"))
                        new_clip_item.append(empty_file)
                
                    new_clip_item.find("start").text = str(start_frame+still_frame[0])
                    new_clip_item.find("end").text = str(start_frame+still_frame[1])
                    new_clip_item.find("in").text = str(still_frame[0])
                    new_clip_item.find("out").text = str(still_frame[1])
                    new_clip_item.find("pproTicksIn").text = str(ticks_in_frame*still_frame[0])
                    new_clip_item.find("pproTicksOut").text = str(ticks_in_frame*still_frame[1])
                    
                    new_clip_item.find("./labels/label2").text = "Rose"
                    new_clip_item.attrib["id"] = "clipitem-"+str(clipitems+id+index+1)
                    
                    stills_track.append(new_clip_item)      
                
                file_data_saved = False
                start_frame += total_frames
    
            new_sequence.find("./media/video").append(movements_track)
            new_sequence.find("./media/video").append(stills_track)
        
        root.append(new_sequence)
        
        result_file = self.project_xml_file[:self.project_xml_file.rindex(".")] + "_clipped.xml"
        self.tree.write(result_file)
        
        self.finishedUpdated.emit(1)
        
    
    def remove_clips(self, root):
        if root.getchildren() is not None:
            for child in root.getchildren():
                if child.tag == 'asset-clip':
                    root.remove(child)
                else:
                    self.remove_clips(child)

    def getFilePathUrl(self, assets, asset_ref):
        for asset in assets:
            if asset.attrib["id"] == asset_ref:
                return asset.attrib["src"]



    @pyqtSlot()
    def process_fcpx(self, show_detection=False, min_area=500, alpha=0.2, threshold=(32, 255), width=1000,  minMotionFrames=5, minNonMotionFrames=5, nonMotionBeforeStart=12, nonMotionAfter=0, minFramesToKeep=35):
        root = self.tree.getroot()
        
        formats = root.findall("./resources/format")
        #xmlstr = ET.tostring(formats[0], encoding='utf8', method='xml')
        #print(xmlstr)

        frameDuration = formats[0].attrib["frameDuration"]
        for i in range(1,len(formats)):
            if formats[i].attrib["frameDuration"] != frameDuration:
                return
                
        frameMod = int(frameDuration.split('/')[0])
        frameDiv = frameDuration.split('/')[1]
        frame_div_int = int(frameDiv[:len(frameDiv)-1])

        fps = int(frameDiv[:len(frameDiv)-1]) / frameMod + 1
        
        new_fcpxml = copy.deepcopy(root)
        asset_clips = root.findall("./library/event/project/sequence/spine/asset-clip")
              
        guid = str(uuid.uuid4())
        new_fcpxml.find("./library/event/project").attrib["uid"] = guid
        project_name = new_fcpxml.find("./library/event/project").attrib["name"]
        new_fcpxml.find("./library/event/project").attrib["name"] = project_name + "_clipped"
        self.remove_clips(new_fcpxml)

        assets = root.findall("./resources/asset")
    
        for index, asset_clip in enumerate(asset_clips):
        
            track = copy.deepcopy(asset_clip)
            track.attrib["name"] = "Checkerboard"

            duration_str = track.attrib["duration"]
            asset_offset_str = track.attrib["offset"]
            
            if "start" not in track.keys():
                asset_start_str = "0s"
            else:
                asset_start_str = track.attrib["start"]
                
            print(asset_start_str)

            if "/" in duration_str:
                bottom = track.attrib["duration"].split('/')[1]
                multiplier = 1
                if bottom != frameDiv:
                    multiplier = frame_div_int / int(bottom[:len(bottom)-1])

                total_duration = int(duration_str.split('/')[0])*multiplier
            else:
                total_duration = int(duration_str[:len(duration_str)-1])*frame_div_int

            if "/" in asset_offset_str:
                
                bottom = track.attrib["offset"].split('/')[1]
                multiplier = 1
                if bottom != frameDiv:
                    multiplier = frame_div_int / int(bottom[:len(bottom)-1])

                asset_offset = int(asset_offset_str.split('/')[0])*multiplier
            else:
                asset_offset = int(asset_offset_str[:len(asset_offset_str)-1])*frame_div_int
                
            if "/" in asset_start_str:
                
                bottom = asset_start_str.split('/')[1]
                multiplier = 1
                if bottom != frameDiv:
                    multiplier = frame_div_int / int(bottom[:len(bottom)-1])

                asset_start = int(asset_start_str.split('/')[0])*multiplier
            else:
                asset_start = int(asset_start_str[:len(asset_start_str)-1])*frame_div_int

            print(track.items())
            asset_ref = track.attrib["ref"]
            file_path_url = self.getFilePathUrl(assets, asset_ref)
            file_path = unquote(file_path_url)
            file_path = file_path[7:]
            file_name = file_path[file_path.rindex("/")+1:]
            msg = "Processing file " + file_name + " ... (" + str(index+1) +"/" + str(len(asset_clips)) + ")"
            print(msg)


            start_frame = 0
            self.progressTextUpdated.emit(msg)
            total_frames, movements, stills, stopped = self.detect_movement(file_path, show_detection, min_area, alpha, threshold, width, minMotionFrames, minNonMotionFrames, nonMotionBeforeStart, nonMotionAfter, minFramesToKeep)
                
            if stopped:
                return

            last_stills = stills[len(stills) - 1][1] - movements[len(movements) - 1][1] > 0
            spine_path = "./library/event/project/sequence/spine"
                                    
            gap = ET.Element('gap')
            gap.attrib["name"] = "Gap"
            gap.attrib["offset"] = "0s"
            duration = stills[0][1]*frameMod
            gap.attrib["duration"] = str(duration)+"/"+frameDiv
            gap.attrib["start"] = "3600s"

            new_asset_clip = copy.deepcopy(track)
            new_asset_clip.attrib["offset"] = "3600s"
            new_asset_clip.attrib["duration"] = str(duration)+"/"+frameDiv
            new_asset_clip.attrib["start"] = str(asset_start)+"/"+frameDiv
            new_asset_clip.attrib["lane"] = "1"

            gap.append(new_asset_clip)
            new_fcpxml.find(spine_path).append(gap)

            current_offset = duration
            current_start = duration
        
            if last_stills:
                _range = len(movements)
            else:
                _range = len(movements) - 1

            for i in range(_range):
                new_asset_clip = copy.deepcopy(track)
                duration = (movements[i][1] - movements[i][0])*frameMod
                new_asset_clip.attrib["offset"] = str(current_offset)+"/"+frameDiv
                new_asset_clip.attrib["duration"] = str(duration)+"/"+frameDiv
                new_asset_clip.attrib["start"] = str(asset_start+current_start)+"/"+frameDiv
            
                current_offset += duration
                current_start += duration
            
                new_fcpxml.find(spine_path).append(new_asset_clip)

                gap = ET.Element('gap')
                gap.attrib["name"] = "Gap"
                gap.attrib["offset"] = str(current_offset)+"/"+frameDiv
                duration = (stills[i+1][1]-stills[i+1][0])*frameMod
                gap.attrib["duration"] = str(duration)+"/"+frameDiv
                gap.attrib["start"] = "3600s" #str(asset_start)+"/"+frameDiv

                new_asset_clip = copy.deepcopy(track)
                new_asset_clip.attrib["offset"] = "3600s"
                new_asset_clip.attrib["duration"] = gap.attrib["duration"]
                new_asset_clip.attrib["start"] = str(asset_start+current_offset)+"/"+frameDiv
                new_asset_clip.attrib["lane"] = "1"

                gap.append(new_asset_clip)
                new_fcpxml.find(spine_path).append(gap)

                current_offset += duration
                current_start += duration

            if not last_stills:
                new_asset_clip = copy.deepcopy(track)
                duration = (movements[len(movements) - 1][1]  - movements[len(movements) - 1][0])*frameMod
                new_asset_clip.attrib["offset"] = str(current_offset)+"/"+frameDiv
                new_asset_clip.attrib["duration"] = str(duration)+"/"+frameDiv
                new_asset_clip.attrib["start"] = str(asset_start+current_start)+"/"+frameDiv
                new_fcpxml.find(spine_path).append(new_asset_clip)

        
        #root.append(new_fcpxml)
        self.tree._setroot(new_fcpxml)
        
        result_file = self.project_xml_file[:self.project_xml_file.rindex(".")] + "_clipped.fcpxml"
        self.tree.write(result_file)
        
        self.finishedUpdated.emit(1)


if __name__ == '__main__':
    
    #alpha = 0.2
    ## construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--file", help="path to the video file")
    
    #ap.add_argument("-v", "--video", help="path to the video file")
    #ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
    
    args = vars(ap.parse_args())
    project_xml_file = args["file"]
    
    clipper = MotionClipper(project_xml_file)
    
    clipper.process(show_detection=False, min_area=100, alpha=0.2, threshold=(5, 255), width=0, minMotionFrames=5, minNonMotionFrames=5, nonMotionBeforeStart=12, nonMotionAfter=0, minFramesToKeep=35)

    
