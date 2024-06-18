# Emotion_Detection_Using_Kaggle_And_IN_Amharic_Languge

To ensure this code runs successfully, several requirements need to be met, including installing the necessary libraries, having the required models and files, and configuring the environment properly. Here are the detailed requirements:

# 1. Install Necessary Libraries

Ensure you have the following Python libraries installed. You can install them using

pip:

- pip install opencv-python-headless
- pip install numpy
- pip install pillow
- pip install tensorflow
- pip install keras

# 2. File Requirements

Make sure the following files are available and their paths are correctly specified in the code:

Haar Cascade Classifier XML: haarcascade_frontalface_default.xml
Pre-trained Emotion Detection Model: model.h5
Font Supporting Amharic: Tera-Regular.ttf

# 3. File Paths

Update the paths in the code to match the location of your files. For example:

python
Copy code
face_classifier = cv2.CascadeClassifier(r'path_to_haarcascade/haarcascade_frontalface_default.xml')
classifier = load_model(r'path_to_model/model.h5')
font_path = r"path_to_font/Tera-Regular.ttf"

# 4. Environment Setup

Webcam Access: Ensure that your system's webcam is accessible. The code uses cv2.VideoCapture(0) to access the default webcam.
Display: Ensure your environment supports GUI operations required for displaying the OpenCV window.

# 5. Additional Considerations

Python Version: Use Python 3.6 or higher.
Dependencies Compatibility: Ensure that the versions of the libraries you install are compatible with each other.

Data Set Link - https://www.kaggle.com/jonathanoheix/face-expression-recognition-dataset
