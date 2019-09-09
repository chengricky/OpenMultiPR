# Open Multimodal Place Recognition

## How to run the code?
Prepare the [environment](#Environment) and the [dependencies](#Dependencies) on your computer, then download the source code, the [additional files](#Additional-Files) and the [dataset](#Dataset). Before running the code, you may change the settings in the [configuration file](#Configuration-File).

\
![The GUI of OpenMPR.](GUI.PNG)

## Related Paper
If you are using this code in your research, please cite the paper:\
Cheng, Ruiqi, et al. "**OpenMPR: Recognize places using multimodal data for people with visual impairments.**" Measurement Science and Technology (2019). https://doi.org/10.1088/1361-6501/ab2106

## Environment
Visual Studio 2017 on Windows 10

## Dependencies
*[OpenCV 4.0 (64 bit)](https://github.com/opencv/opencv.git)*: set OpenCV path as OpenCV_DIR in system environments\
*[DBoW3](https://github.com/rmsalinas/DBow3.git)*: set DBoW3 path as DBoW3_DIR in system environments\
*[FFTW3](http://fftw.org/download.html)*: set FFTW path as FFTW_DIR in system environments

## Additional Files
The pre-trained model of CNN could be downloaded at [GoogLeNet-Places365](https://drive.google.com/file/d/1bB4eIGdq63UHZJBKOqL2PKHrSOuJ3rRy/view?usp=sharing), which should be unzipped to the source code folder.

## Dataset
The dataset is available at [Multimodal Dataset](https://drive.google.com/file/d/1NuRUaZA_g0rBzJXYLqy4RlgZw7OGDvnv/view?usp=sharing).

## Configuration File
The configuration file `Config.yaml` is in the folder of `OpenMultiPR`. The detailed information of the parameters could be found in `Config.yaml`. 

In the yaml file, the dataset and BoW vocabulary paths are assigned. The configuration files also includes the parameters on wheter to use the specific modalities (i.e. RGB, Infrared, Depth and GNSS data) and the corresponding descriptors (i.e. GIST, ORB-BoW, LDB). The running mode could be set in the file.

