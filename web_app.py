import streamlit as st
import os
import pandas as pd
import emotion
import plotly.graph_objects as go

st.title('Emotion Detection from Music')
st.write('Developed by Team BAAS')

song = st.file_uploader("Choose song..", type="mp3")

if st.button('Calculate Emotion'):
    path = os.path.join('temp_in',song.name)
    st.write(song.name)
    with open(os.path.join('temp_in',song.name),'wb') as f:
        f.write(song.getbuffer())

    audio_bytes = song.read()
    aud = st.audio(audio_bytes, format='audio/mp3')
    #duration = st.text_input('Duration (in seconds)',90)
    pp5 = emotion.get_emotion(path,duration=90)
    prediction = pp5

    #plotly
    fig = go.Figure()
    fig.add_trace(go.Indicator(
    value = int(prediction[0][0]*100),
    title = {'text': "Peace"},
    gauge = {'axis': {'range': [None, 100]}},
    domain = {'row': 0, 'column': 0}))

    fig.add_trace(go.Indicator(
    value = int(prediction[0][1]*100),
    title = {'text': "Sadness"},
    gauge = {'axis': {'range': [None, 100]}},
    domain = {'row': 0, 'column': 1}))

    fig.add_trace(go.Indicator(
    value = int(prediction[0][2]*100),
    title = {'text': "Disturbed"},
    gauge = {'axis': {'range': [None, 100]}},
    domain = {'row': 0, 'column': 2}))

    fig.add_trace(go.Indicator(
    value = int(prediction[0][3]*100),
    title = {'text': "Excitement"},
    gauge = {'axis': {'range': [None, 100]}},
    domain = {'row': 0, 'column': 3}))

    fig.update_layout(
    grid = {'rows': 1, 'columns': 4, 'pattern': "independent"},
    template = {'data' : {'indicator': [{
        'mode' : "number+gauge",
       }] }})

    st.title('Emotion Analysis')
    st.plotly_chart(fig)

    st.success('Success!!')
