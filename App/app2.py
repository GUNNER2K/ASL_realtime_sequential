import streamlit as st

from scripts.capture_script import *

from tensorflow.keras.models import load_model



# @st.cache_resource
# def load_model():
#     model = load_model('App/assets/asl_model_2.h5')
#     return model

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
        st.markdown(' - Mediapipe')
        st.markdown(' - Streamlit')
        st.markdown(' - Matplotlib')

    
    with col2:
        st.image('App/assets/asl.jpg')

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dataset Description")
        st.write('''The Dataset we are using is an image dataset consisting of around 2000 images across 36 different classes ranging from A-Z and 0-9.
                 
                 Dataset Link: https://www.kaggle.com/datasets/ayuraj/asl-dataset
                 
                 
This Dataset was preprocessed and Augmented and was used to train our model on.''')
        st.text("")
        st.text("")
        st.text("")
        imgb = cv2.imread('App/assets/class_distribution.png')
        imgb = cv2.cvtColor(imgb, cv2.COLOR_BGR2RGB)      
        st.image(imgb)  

    with col2:
        st.subheader("Dataset Sample Images")
        imga = cv2.imread('App/assets/example_ds.png')
        imga = cv2.cvtColor(imga, cv2.COLOR_BGR2RGB)
        
        st.image(imga)


with tab3:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model Description")
        st.write('''As we have all image data, the best way to extract meaningful information from the data is using Convolutional Neural Networks(CNN). 
                 
The model we created for this application is a 3-Block CNN with each block containing a Convolution Layer, a Maxpooling Layer and a Dropout Layer.

You can see the Model architechture in the following Diagram.''')

    with col2:
        st.subheader("Model Architecture")
        img = cv2.imread('App/assets/model.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img)

with tab4:
    model_path = 'App/assets/asl_model_3.h5'
    # model = load_model(model_path)

    col1, col2 = st.columns(2)
    word = ''
    with col1:
        st.subheader('To start the translation, click on the "Start Translator" button, It will then access your camera')
        if st.button('Start Translator', key= 'start_button'):
            frame_placeholder = st.empty()
            model = load_model(model_path)
            #word = ''
            cap = cv2.VideoCapture(0)
            frame_counter = 0
            stop = st.button('Stop', key= 'stop_button')
            while cap.isOpened():
                frame_counter += 1
                ret, frame = cap.read()

                frame = cv2.flip(frame, 1)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                frame , coord = draw_hands(frame)
                if frame_counter == 60:
                    if coord:
                        letter= draw_prediction(frame, coord, model)
                        word = word+ letter
                        frame_counter = 0
                    else:
                        frame_counter = 0
                        st.empty()
                        st.write(word)
                        word = ''
                if not ret:
                    st.write('Video Capture has ended.')
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame_placeholder.image(frame, channels='BGR')

                if stop:
                    st.empty()
                    break
                #st.write(prediction)
            cap.release()
            cv2.destroyAllWindows()
    with col2:
        st.write(word)    
        