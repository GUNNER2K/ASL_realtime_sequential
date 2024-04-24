import streamlit as st
import numpy as np
import os
import string
import mediapipe as mp
import cv2
from my_functions import *
import keyboard
from tensorflow.keras.models import load_model
import language_tool_python

# @st.cache_resource
# def load_model():
#     model = load_model('App/assets/asl_model_2.h5')
#     return model
# Set the path to the data directory
PATH = os.path.join('data')

# Create an array of action labels by listing the contents of the data directory
actions = np.array(os.listdir(PATH))

# Load the trained model
model = load_model('my_model')

# Create an instance of the grammar correction tool
tool = language_tool_python.LanguageToolPublicAPI('en-UK')

# Initialize the lists
sentence, keypoints, last_prediction, grammar, grammar_result = [], [], [], [], []

st.set_page_config(layout='wide')



st.header('ASL Translator App')

tab1, tab2, tab3, tab4 = st.tabs(['Home', 'Dataset', 'Model', 'Translator'])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('About the App')
        st.write('''This app is a Sign Language to text translator app for people who dont know how to use sign language.
             I'm using the American Sign Language standards for this purpose as it is the most widely used sign language standard out there.''')
        st.write('The App uses a Convolutional Neural Network(CNN) built and trained with the help of Tensorflow to detect and recognize the alphabet that is being spelled by hand.')
        st.markdown('**Techstack used for the app:**')
        st.markdown(' - Tensorflow')
        st.markdown(' - Pandas')
        st.markdown(' - Numpy')
        st.markdown(' - Matplotlib')

    
    with col2:
        st.image('App/assets/asl.jpg')

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dataset Description")

    with col2:
        st.subheader("Dataset Sample Images")

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model Description")

    with col2:
        st.subheader("Model Architecture")

with tab4:
    # model_path = 'App/assets/asl_model_2.h5'
    # model = load_model(model_path)

    col1, col2 = st.columns(2)
    word = ''
    with col1:
        if st.button('Start Translator', key= 'start_button'):
            frame_placeholder = st.empty()
            #word = ''
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.write("Cannot access camera.")
                exit()
            stop = st.button('Stop', key= 'stop_button')
            with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
                # Run the loop while the camera is open
                while cap.isOpened():
                    # Read a frame from the camera
                    _, image = cap.read()
                    # Process the image and obtain sign landmarks using image_process function from my_functions.py
                    results = image_process(image, holistic)
                    # Draw the sign landmarks on the image using draw_landmarks function from my_functions.py
                    image = draw_landmarks(image, results)
                    # Extract keypoints from the pose landmarks using keypoint_extraction function from my_functions.py
                    keypoints.append(keypoint_extraction(results))

                    # Check if 10 frames have been accumulated
                    if len(keypoints) == 20: #10 
                        # Convert keypoints list to a numpy array
                        keypoints = np.array(keypoints)
                        # Make a prediction on the keypoints using the loaded model
                        prediction = model.predict(keypoints[np.newaxis, :, :])
                        # Clear the keypoints list for the next set of frames
                        keypoints = []

                        # Check if the maximum prediction value is above 0.9
                        if np.amax(prediction) > 0.9:
                            # Check if the predicted sign is different from the previously predicted sign
                            if last_prediction != actions[np.argmax(prediction)]:
                                # Append the predicted sign to the sentence list
                                sentence.append(actions[np.argmax(prediction)])
                                # Record a new prediction to use it on the next cycle
                                last_prediction = actions[np.argmax(prediction)]

                    # Limit the sentence length to 7 elements to make sure it fits on the screen
                    if len(sentence) > 7:
                        sentence = sentence[-7:]

                    # Reset if the "Spacebar" is pressed
                    if keyboard.is_pressed(' '):
                        sentence, keypoints, last_prediction, grammar, grammar_result = [], [], [], [], []

                    # Check if the list is not empty
                    if sentence:
                        # Capitalize the first word of the sentence
                        sentence[0] = sentence[0].capitalize()

                    # Check if the sentence has at least two elements
                    if len(sentence) >= 2:
                        # Check if the last element of the sentence belongs to the alphabet (lower or upper cases)
                        if sentence[-1] in string.ascii_lowercase or sentence[-1] in string.ascii_uppercase:
                            # Check if the second last element of sentence belongs to the alphabet or is a new word
                            if sentence[-2] in string.ascii_lowercase or sentence[-2] in string.ascii_uppercase or (sentence[-2] not in actions and sentence[-2] not in list(x.capitalize() for x in actions)):
                                # Combine last two elements
                                sentence[-1] = sentence[-2] + sentence[-1]
                                sentence.pop(len(sentence) - 2)
                                sentence[-1] = sentence[-1].capitalize()

                    # Perform grammar check if "Enter" is pressed
                    if keyboard.is_pressed('enter'):
                        # Record the words in the sentence list into a single string
                        text = ' '.join(sentence)
                        # Apply grammar correction tool and extract the corrected result
                        grammar_result = tool.correct(text)

                    image = np.int32(image)

                    if grammar_result:
                        # Calculate the size of the text to be displayed and the X coordinate for centering the text on the image
                        textsize = cv2.getTextSize(grammar_result, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                        text_X_coord = (image.shape[1] - textsize[0]) // 2

                        # Draw the sentence on the image
                        cv2.putText(image, grammar_result, (text_X_coord, 470),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    else:
                        # Calculate the size of the text to be displayed and the X coordinate for centering the text on the image
                        textsize = cv2.getTextSize(' '.join(sentence), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                        text_X_coord = (image.shape[1] - textsize[0]) // 2

                        # Draw the sentence on the image
                        cv2.putText(image, ' '.join(sentence), (text_X_coord, 470),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    # Show the image on the display
                    frame_placeholder.image(image, channels='BGR')

                    cv2.waitKey(1)

                    if stop:
                        st.empty()
                        break
                # Release the camera and close all windows
                cap.release()
                cv2.destroyAllWindows()

                # Shut off the server
                tool.close()
    with col2:
        st.write(word)    
        