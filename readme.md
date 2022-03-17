# TUM x DLR project

A helper tool that automatically detects, shows, and fits a polynomial to aponeurosis in ultrasound images of the calf muscle.

Project structure:

- Edge Detection: first approach to extract information regarding the aponeurosis
- gui: graphic user interface of the detections, a docker based application
- matlab_converter: a tool that converts .dcm medical images to .png formats
- multi_detection: preprocessing of the images and annotations, training of the model, and prediction/detection of the aponeurosis
