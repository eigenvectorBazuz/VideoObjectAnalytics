from single_movie_analysis import discover_objects_in_video
from reid import Encode, CompareTrackList
from utils import annotate_video_with_tracks

def analyze_and_annotate_videos(videos_list):
  results = []
  for video in videos_list:
    tracks = discover_objects_in_video(video, 'yolo12x.pt', tie_point_params={'video_chunk_size':50, 'overlap':5, 'grid_size':20}, 
                               include_final_split=False, return_data=False)
    results.append(tracks)


