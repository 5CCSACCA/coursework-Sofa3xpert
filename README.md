# YOLO setup

This project allows you to run a YOLOv5 model for image classification.

Do the following in terminal:
    
install required dependencies

    pip install -r requirements.txt

Launch the script by:

    python classification.py path/to/image.jpg
    
## Weights folder

Transfer any custom or newest weights in this folder

## main.py 

Responsible for FastAPI setup, including '/status', '/version', '/predict'.

## Dockerfile 

First version of model ready for deployment

## index.html 

Basic version of front-end 
