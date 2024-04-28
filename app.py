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
        st.write('''This app is a Sign Language to text translator app for created by Group 14 of BAI 20 Branch for Capstone Project DSN 4095 and DSN 4096.
             we are using the American Sign Language standards for this purpose as it is the most widely used sign language standard out there.''')
        st.write('The App uses a stacked LSTM network built and trained with the help of Tensorflow to detect sequences of data. Here RNN is used instead of CNN to provide room for contextualization and real life application.')
        st.write('We would like to thank our guide, `DR. ANIL KUMAR YADAV` for guiding us for the entirity of this Capstone Project DSN 4095 and DSN 4096')
        st.markdown('**Techstack used for the app:**')
        st.markdown(' - Tensorflow: For building and training the model')
        st.markdown(' - Pandas: Standard ML library')
        st.markdown(' - Numpy: Standard ML library for mathematical calculations')
        st.markdown(' - Matplotlib: Important visuals')
        st.markdown(' - Mediapipe: Tracking of hands and landmarks')
        st.markdown(' - Open-Cv: For Accessing web camera for real time translation')

    
    with col2:
        st.image('App/assets/asl.jpg')

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dataset Description")
        st.write('''The data used to train the model is a custom dataset that has been aquired through a proficient data collection process.
                 the data was collected by capturing n number of sequences for each sign, where n can be given by the user. 
                 Each sequence has 20 frames relevant to the action of the hand sign, from these 20 frames the landmarks were extracted using the mediapipe library and stored in a .npy file
                 instead of the entire image as it is space efficient as well as trains the model in a better and faster way''')
        
        st.write('To the right there are some examples on how these images were collected ')

    with col2:
        st.subheader("Dataset Sample Images")

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model Description")
        st.write('''
This model is a sequential neural network built using the Keras API, specifically designed for sequential data processing, such as time series or sequential pattern recognition. Let's break down each layer:

1. LSTM Layer (32 units):
   - This layer consists of Long Short-Term Memory (LSTM) cells, a type of recurrent neural network (RNN) architecture designed to capture long-term dependencies in sequential data.
   - The layer has 32 LSTM units.
   - `return_sequences=True` indicates that the layer will return the full sequence of outputs rather than just the last output.
   - `activation='relu'` specifies the Rectified Linear Unit (ReLU) activation function, which introduces non-linearity to the model.

2. LSTM Layer (64 units):
   - Another LSTM layer with 64 units.
   - `return_sequences=True` is set to maintain the sequence output.
   - `activation='relu'` as the activation function.

3. LSTM Layer (32 units):
   - Another LSTM layer with 32 units.
   - Unlike the previous layers, `return_sequences=False`, indicating that it will only return the last output of the sequence.
   - `activation='relu'` is used.

4. Dense Layer (32 units):
   - A densely connected layer with 32 units.
   - This layer doesn't specify `return_sequences`, as it's not a recurrent layer.
   - `activation='relu'` is the activation function applied to the outputs.

5. Dense Layer (output layer):
   - Final output layer with a number of units equal to the number of actions (assuming `actions.shape[0]` represents the number of different actions).
   - `activation='softmax'` is used here, which is typical for classification tasks. It outputs a probability distribution over the different classes, indicating the likelihood of each action being the correct one.

Overall, this model architecture seems to be designed for sequential data input with a specific focus on capturing temporal dependencies using multiple LSTM layers followed by some dense layers for classification or prediction tasks.
''')

    with col2:
        st.subheader("Model Architecture")
        st.image('App/assets/LSTM.png')

with tab4:
    # model_path = 'App/assets/asl_model_2.h5'
    # model = load_model(model_path)
    st.subheader(' This is Main Translation Page')
    st.write('This is where you will be able to translate sign language into texts in real time')
    st.write('Follow the instructions on the right side to use the app -->')
    col1, col2 = st.columns(2)
    word = ''

    with col2:
        st.subheader('Instructions: ')
        st.markdown(' - Click on "Start Translator" button to Start the app')
        st.markdown(' - It will request permission to access your webcam')
        st.markdown(' - The translated text will be displayed on the screen')
        st.markdown(' - To stop the translation, click on "Stop" Button ')
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
