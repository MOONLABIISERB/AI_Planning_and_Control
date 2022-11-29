# AI Planning and Control

Autonomous vehicle control with CARLA using Learning by Cheating (two front facing camera system)

## Installation

### Conda Environment 
Create a conda environment with Python 3.7
```bash
conda create --name carla python=3.7
```

Activate the conda environment
```bash
conda activate carla
```

### Clone the Github Repository
Clone the github repository
```bash
git clone https://github.com/MOONLABIISERB/AI_Planning_and_Control.git
```
### Install Dependencies
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install dependencies listed in ```requirements.txt```.

```bash
pip install -r requirements.txt
```
### Download Checkpoints
Download the model checkpoints from [google drive](https://drive.google.com/drive/folders/1X_JNFwbyAQEO3Ep6KqFAFahhv0MTs_N1?usp=share_link) to the root folder.



## Usage

Open a terminal and run CARLA Server
```bash
./CarlaUE4.sh -RenderOffScreen -fps=10 -benchmark
```
cd to <path/to/root>/PythonAPI/examples
```bash
cd <path/to/root>/PythonAPI/examples
python auto_drive_new.py --agent=LBC
```
## Results
[![Click to see the simulation recording](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://drive.google.com/file/d/1Ty72Izjl-0J2bmEhZfh9lciKeHhFPz6Q/view?resourcekey)
