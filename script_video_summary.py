from video_summary import VideoChatBot

clip_file = '/content/drive/MyDrive/BR/seg0.mp4'

vcb = VideoChatBot(size = "0.5B", quant_4bit = True, target_hw = (360, 480), frames_per_chunk = 30)

user_prompt = """ Identify key events in the segment (e.g., object appearances, interactions, unusual movements, bridges ... ), in particular vehicle maneuvres.
"""
report = vcb(clip_file, user_prompt)

print(report)
