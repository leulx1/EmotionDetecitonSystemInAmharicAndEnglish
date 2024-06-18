import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Load the pre-trained face detection model
face_classifier = cv2.CascadeClassifier(r'D:\Users\leul\Ai\Emotion-Detection-System-AI\Emotion_Detection_CNN-main\haarcascade_frontalface_default.xml')
try:
    # Load the pre-trained emotion detection model
    classifier = load_model(r'D:\Users\leul\Ai\Emotion-Detection-System-AI\Emotion_Detection_CNN-main\model.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load model: {e}")
    exit()

# Define emotion labels in English and Amharic
english_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
amharic_labels = ['ቁጣ', 'ጠላ', 'ፍርሃት', 'ደስተኛ', 'ኖርማል', 'መከፋት', 'መገረም']

# Initialize video capture from the webcam
cap = cv2.VideoCapture(0)

# Load the Amharic-supporting font
font_path = "D:\\Users\\leul\\Ai\\Emotion-Detection-System-AI\\Emotion_Detection_CNN-main\\Tera-Regular.ttf"
try:
    font = ImageFont.truetype(font_path, 45, encoding='unic')  # Load font with size 45 and unic encoding
    print("Font loaded successfully.")
except IOError:
    print("Font file not found or failed to load. Please check the path to the font.")
    exit()
except Exception as e:
    print(f"Error loading font: {e}")
    exit()

# Create a named window
cv2.namedWindow('Emotion Detector', cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        print("Failed to capture frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale
    faces = face_classifier.detectMultiScale(gray)  # Detect faces in the frame

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]  # Extract the region of interest (ROI) for the face
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)  # Resize ROI to 48x48

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0  # Normalize the ROI
            roi = img_to_array(roi)  # Convert the ROI to an array
            roi = np.expand_dims(roi, axis=0)  # Expand dimensions to match model input

            prediction = classifier.predict(roi)[0]  # Predict emotion
            label_index = prediction.argmax()  # Get the index of the highest prediction
            english_label = english_labels[label_index]  # Get the corresponding English label
            amharic_label = amharic_labels[label_index]  # Get the corresponding Amharic label
            print(f"Predicted label: {english_label} {amharic_label}")

            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            
            # Draw English label above the rectangle
            cv2.putText(frame, english_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Draw Amharic label below the rectangle using PIL
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB
                pil_image = Image.fromarray(frame_rgb)  # Create a PIL image from the frame
                draw = ImageDraw.Draw(pil_image)  # Create a drawing context
                
                # Calculate position for the Amharic label
                amharic_label_size = draw.textbbox((0, 0), amharic_label, font=font)
                amharic_label_width = amharic_label_size[2] - amharic_label_size[0]
                amharic_position = (x + (w - amharic_label_width) // 2, y + h + 10)
                
                # Draw the Amharic label on the image
                draw.text(amharic_position, amharic_label, font=font, fill=(0, 155, 0))

                # Convert back to BGR for OpenCV
                frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"Error rendering Amharic label: {e}")
                cv2.putText(frame, amharic_label, (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 155, 0), 2)  # Fallback to Amharic label
        else:
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Display 'No Faces' if no face is detected

    cv2.imshow('Emotion Detector', frame)  # Display the resulting frame

    # Check for key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Exit the loop when 'q' is pressed
        break
    elif key == ord('f'):  # Maximize the window when 'f' is pressed
        cv2.setWindowProperty('Emotion Detector', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    elif key == ord('m'):  # Minimize the window when 'm' is pressed
        cv2.setWindowProperty('Emotion Detector', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

cap.release()  # Release the video capture object
cv2.destroyAllWindows()  # Close all OpenCV windows
