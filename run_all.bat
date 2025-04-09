@echo off
python process_pose_json.py holistic_json/holistic_label00_standing.json 0
python process_pose_json.py holistic_json/holistic_label01_reading.json 1
python process_pose_json.py holistic_json/holistic_label02_behind.json 2
pause
