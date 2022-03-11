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
from PyQt6 import QtCore
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, QMutex

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
    
    def detect_movement(self, filename, show_detection=False, min_area=500, alpha=0.2, threshold=(32, 255), width=1000):
        print("Parameters passed:")
        print(show_detection, min_area, alpha, threshold, width)
        #fvs = FileVideoStream(filename).start()
        
        video = cv2.VideoCapture(filename)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        print(num_frames)
        # initialize the first frame in the video stream
        firstFrame = None
        
        frame_number = 0
        current_state = "Unoccupied"
        
        stills = []
        movements = []
        
        start = 0
        end = 0
        
        # loop over the frames of the video
        for frame_number in tqdm(range(num_frames)):
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
            (grabbed, frame) = video.read()
            
            #frame = fvs.read()
            
            # if the frame could not be grabbed, then we have reached the end
            # of the video
        
            # resize the frame, convert it to grayscale, and blur it
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
                current_state = "Unoccupied"
                end = frame_number
                movements.append((start, end))
        
                start = frame_number
        
            # loop over the contours
            for c in cnts:
                # if the contour is too small, ignore it
                if cv2.contourArea(c) < min_area:
                    continue
        
                if current_state == "Unoccupied":
                    end = frame_number
                    if end > 0:
                        stills.append((start, end))
        
                    start = frame_number
                    current_state = "Occupied"
        
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
            
            if frame_number == num_frames - 1:
                if current_state == "Occupied":
                    print((start, frame_number - 1))
                    movements.append((start, frame_number - 1))
                else:
                    print((start, frame_number - 1))
                    stills.append((start, frame_number - 1))
            
                    
        print("With movements: ")
        print(movements)
        
        print("Without movements: ")
        print(stills)  
            
        # cleanup the camera and close any open windows
        video.release()
        
        if show_detection:
            cv2.destroyWindow("Thresh")
            cv2.destroyWindow("Frame Delta")
            cv2.destroyWindow("Motion Detection Feed")
        
        return frame_number, movements, stills, False
    
    def remove_tracks(self, root):
        if root.getchildren() is not None:
            for child in root.getchildren():
                if child.tag == 'track':
                    root.remove(child)
                else:
                    self.remove_tracks(child)
    
    @pyqtSlot()
    def process(self, show_detection=False, min_area=500, alpha=0.2, threshold=(32, 255), width=1000):
        root = self.tree.getroot()
        
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
                total_frames, movements, stills, stopped = self.detect_movement(file_path, show_detection, min_area, alpha, threshold, width)
                
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
    
    clipper.process(show_detection=False, min_area=500, alpha=0.2, threshold=(32, 255), width=1000)

    