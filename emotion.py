#Load major libraries
import pandas as pd
import numpy as np
import essentia.standard
from essentia.standard import *
import pickle
import librosa
import os
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report

#Load Scaler for Pre-Processing
from sklearn.preprocessing import StandardScaler
with open(r"Model_1/scaler_1.pickle", 'rb') as h1:
    scaler_1 = pickle.load(h1)

#Load Random Forest Model
from sklearn.ensemble import RandomForestClassifier
with open(r'Model_1/rf_model_75.pickle', 'rb') as h2:
    clf_rf = pickle.load(h2)


def extract_features(songname,duration):
  y, sr = librosa.load(songname, mono=True, duration=duration)
  #loader = essentia.standard.MonoLoader(filename=y)
  audio = y
  energy_obj = essentia.standard.Energy()
  energy = energy_obj(audio)
  dance_obj = essentia.standard.Danceability()
  danceability = dance_obj(audio)
  loud_obj = essentia.standard.Loudness()
  loudness = loud_obj(audio)
  chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
  spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
  spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
  rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
  zcr = librosa.feature.zero_crossing_rate(y)
  mfcc = librosa.feature.mfcc(y=y, sr=sr)
  mfccs = [np.mean(x) for x in mfcc]
  data_temp = {
    'chroma_stft':np.mean(chroma_stft),
    'centroid': np.mean(spec_cent),
    'bandwidth': np.mean(spec_bw),
    'rolloff': np.mean(rolloff),
    'zero_crossing_rate': np.mean(zcr),
    'mfcc_0': mfccs[0],
    'mfcc_1': mfccs[1],
    'mfcc_2': mfccs[2],
    'mfcc_3': mfccs[3],
    'mfcc_4': mfccs[4],
    'mfcc_5': mfccs[5],
    'mfcc_6': mfccs[6],
    'mfcc_7': mfccs[7],
    'mfcc_8': mfccs[8],
    'mfcc_9': mfccs[9],
    'mfcc_10': mfccs[10],
    'mfcc_11': mfccs[11],
    'mfcc_12': mfccs[12],
    'mfcc_13': mfccs[13],
    'mfcc_14': mfccs[14],
    'mfcc_15': mfccs[15],
    'mfcc_16': mfccs[16],
    'mfcc_17': mfccs[17],
    'mfcc_18': mfccs[18],
    'mfcc_19': mfccs[19],
    'danceability': danceability[0],
    'loudness':loudness,
    'energy':energy
  }
  return data_temp


def get_emotion(song,duration=90):
  data = extract_features(song,duration)
  feats = pd.DataFrame(data,index=[0])
  n_feats = np.array(feats.iloc[0])
  feats_new = scaler_1.transform([n_feats])
  skip = []

  try:
    pred_proba5 = clf_rf.predict_proba(feats_new)
    #print('\n RANDOM FOREST CLASSIFIER \n Peace:',int(pred_proba5[0][0]*100),'%','\n','Sadness:',int(pred_proba5[0][1]*100),'%','\n','Disturbed:',int(pred_proba5[0][2]*100),'%',
    #  '\n','Excitement:',int(pred_proba5[0][3]*100),'%')
  except:
    pred_proba5 = [False,False,False,False]
    #print('\n ERROR IN PROCESSING RANDOM FOREST')


  return pred_proba5


  #print('\n ENSEMBLE \n Peace:',int(peace_pred*100),'%','\n Sadness:',int(sad_pred*100),'%','\n Disturbed:',int(anger_pred*100),'%','\n Excitement:',int(dance_pred*100),'%')
