# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 22:28:46 2021

@author: 010 3304 6536
"""


import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os


def trim_audio_data(audio_file,sr=16000,sec=1):
    
    "'오디오파일 1개를 입력한 sec로 나눠주는 함수 sr은 sampling rate, sec는 나눌 구간의 시간"'
    
    y, sr = librosa.load(audio_file, sr=sr)
    
    "'파일을 입력한 구간으로 나누기 전에 구간으로 나누어떨어지도록 남는 뒷부분을 버림"' 

    ny = y[:len(y)-len(y)%(sr*sec)]
    
    "'한 행이 한 구간(sec * sr)인 넘파이 행렬로 변환"' 

    ny=ny.reshape((-1, sr*sec)) 

    return ny 

def prototype(audio_path,sr=16000,sec=1): 

    "'trim_audio_data함수를 for문으로 엮은 함수'" 

    audio_list = os.listdir(audio_path) 

    result=np.zeros((1,sr*sec)); 

    "'audio_path내의 모든 wav파일을 for문으로 trim_audio_data함수에 넣어줌'" 

    for audio_name in audio_list:
        if audio_name.find('wav') is not -1:
            audio_file = audio_path + '/' + audio_name
    
            assist=trim_audio_data(audio_file,sr,sec)
    
'"구간으로 나눠진 파일(넘파이 행렬)은 result에 하나로 쌓음'" 

            result=np.concatenate((result,assist));
    
    result=result[1:,:];
    
    return result 

def Mel_S(vector,frame_length=0.025,frame_stride=0.010,sr=16000,n_y=101): 

"'audio 행렬을 넣으면 mel spectogram으로 변환시켜주는 함수'" 

    # wav_length = len(y)/sr
    input_nfft = int(round(sr*frame_length))
    input_stride = int(round(sr*frame_stride)) 

    S = librosa.feature.melspectrogram(y=vector, n_mels=n_y, n_fft=input_nfft, hop_length=input_stride) 

    return S 

def Logmel_S(vector,frame_length=0.025,frame_stride=0.010,sr=16000,n_y=101): 

"'audio 행렬을 넣으면 log mel spectogram으로 변환시켜주는 함수'" 

    # wav_length = len(y)/sr
    input_nfft = int(round(sr*frame_length))
    input_stride = int(round(sr*frame_stride)) 

    S = librosa.feature.melspectrogram(y=vector, n_mels=n_y, n_fft=input_nfft, hop_length=input_stride)
    S = librosa.power_to_db(S,ref=np.max) 

    return S


def predata(vector,frame_length=0.025,frame_stride=0.010,sr=16000,n_y=101):


#결과값 저장할 빈 list함수 result 정의
    result=[]
    aa=len(vector) 

'"prototype함수 결과로 나오는 행렬에서 한 행 = 한 구간이니 for문으로 매 행을 뽑아내서 Mel_S함수에 넣고 result 리스크에 저장'"
    for i in range(0,aa):
        assist_vector=vector[i,:]
        ex=Mel_S(assist_vector,frame_length,frame_stride,sr,n_y)
        result.append(ex)
    
    return result 

"'predata의 입력은 prototype의 결과로 나온 넘파이 행렬이고 결과값은 list에 mel spectrogram벡터가 저장되어 나오게 됨'" 

def log_mel_predata(vector,frame_length=0.025,frame_stride=0.010,sr=16000,n_y=101): 

'"predata함수를 log_mel spectrogram을 뽑는 함수로 수정한 함수" 

    result=[]
    aa=len(vector) 

    for i in range(0,aa):
        assist_vector=vector[i,:]
        ex=Logmel_S(assist_vector,frame_length,frame_stride,sr,n_y)
        result.append(ex) 

    return result
