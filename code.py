from ultralytics import YOLO, YOLOE

# # Load a model
# model = YOLO("yolo11x.pt")  # pretrained YOLO11n model
# model = YOLOE("yoloe-11l-seg-pf.pt")
# Initialize a YOLOE model
# model = YOLOE("yoloe-11l-seg.pt")  # or select yoloe-11s/m-seg.pt for different sizes
# names = ["person", "vehicle", "bridge", "sign"]
# model.set_classes(names, model.get_text_pe(names))

# clip_file = '/content/drive/MyDrive/BR/seg0.mp4'
# track_results = model.track(clip_file, agnostic_nms=True, persist=True, tracker="/content/drive/MyDrive/BR/botsort_modified.yaml")  # Tracking with default tracker


def temp(n):
  print(n+3)
  print('sd')

# use the BB's
# TBD - use the masks
def get_raw_YOLO_detections(video, yolo_model):
  results = yolo_model.predict(video, agnostic_nms=True)
  return results
  

# video is an mp4 file or some compatible source/iterator
# TBD - print a list of supported models and make a switch case
def discover_objects_in_video(video, yolo_model_name):
  yolo_model = YOLO(yolo_model_name) # unless it's a YOLOE model....
  pass
