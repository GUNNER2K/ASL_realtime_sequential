import streamlit as st

from scripts.capture_script import *

from tensorflow.keras.models import load_model



# @st.cache_resource
# def load_model():
#     model = load_model('App/assets/asl_model_2.h5')
#     return model

st.set_page_config(layout='wide')



st.header('ASL Trasnlator App')

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
                if frame_counter == 20:
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
        