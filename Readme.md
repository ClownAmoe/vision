Create venv
py -m venv venv

Start vev
.\venv\Scripts\Activate.ps1

Intall all libraries
pip install -r .\requirements.txt

if add new lib
pip freeze > requirements.txt

Drone video pipeline (new)
- Quick parser check:
	- d:/Projects/Vision37/vision/venv/Scripts/python.exe drone_demo.py
- Motion estimation (downward-facing drone camera):
	- d:/Projects/Vision37/vision/venv/Scripts/python.exe drone_motion_estimation.py --detectors OPTICAL_FLOW --max_frames 500
	- Use --target_fps 5 or --frame_stride 6 to increase baseline between frames
