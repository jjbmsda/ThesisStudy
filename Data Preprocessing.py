import os #운영체제에서 제공되는 여러 기능을 파이썬에서 이용할 수 있도록 하는 모듈
import librosa #음악과 오디오 분석과 관련된 기능을 제공하는 모듈
from os import listdir #디렉터리에 있는 파일들의 리스트를 구하기 위해
from os.path import isfile, join #파일인지 확인 및 디렉터리와 파일명을 이어주는 기능
import numpy as np #수학 및 과학 연산을 위한 모듈
import scipy.io.wavfile #수학 및 과학 연산을 위한 scipy 모듈 중 특별히 wavfile IO 와 관련된 모듈
import pickle #객체의 형태를 그대로 유지하면서 파일을 저장하고 불러올 수 있게 하는 모듈

ROOT_PATH = os.getcwd()  #현재 디렉토리를 ROOT_PATH에 저장함
PATH_NOISE = ROOT_PATH + '/RIRS_NOISES/pointsource_noises'
PATH_TRAIN = ROOT_PATH + '/data1/vox1/wav'
PATH_FEATURE_OUTPUT = ROOT_PATH + '/output/feature'

def rms(input):
    rms = np.sqrt(np.mean(input**2))
    return rms
 
def equalizingRMS(source,target):
    target = rms(source)/rms(target)*target
    return target
 
def addNoise(speech,noise,snr):
    if len(speech) > len(noise):
       speech = speech[0:len(noise)]
    else :
       noise = noise[0:len(speech)]
    noisy = speech + np.sqrt(np.sum(np.abs(speech)**2))/ np.sqrt(np.sum(np.abs(noise)**2) *
np.power(10,snr*0.1)) * noise
    return noisy
 
# =============================================================================
# Functions that Takes the STFT and EXTRACTs THE LPS FEATURES
# =============================================================================

def framing(y, win_length=400, hop_length=160):
    n_frames = 1 + int((len(y) - win_length) / hop_length)
    y_frames = np.lib.stride_tricks.as_strided(y, shape=(win_length, n_frames),
strides=(y.itemsize, hop_length * y.itemsize))
    return y_frames.T
 
def stft(y, n_fft=512, win_length=400, hop_length=160, window='hann'):
    stft_window = scipy.signal.get_window(window, win_length, fftbins=True)
    frames = framing(y, win_length=win_length, hop_length=hop_length)
    return np.fft.rfft( stft_window * frames , n_fft, axis=1).T
 
def spectrogram(y=None, n_fft=512, win_length=400, hop_length=160, power=1):
    return np.abs(stft(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length))**power
    
# =============================================================================
# Read Noise Files in PATH_NOISE Folder (with normalized to standard waveform)
# =============================================================================
folder_list = os.listdir(PATH_TRAIN)

for folder in folder_list:
    fpath = librosa.util.find_files(PATH_TRAIN+'/'+folder)
    
standard_wav,rate = librosa.load(fpath[0], sr=None)

noise_waves=[] #noise 파일의 값들을 저장해두는 리스트

for noiseFile in [f for f in listdir(PATH_NOISE) if f.endswith(".wav")]:
    noise_wav,rate = librosa.load(PATH_NOISE+'/'+noiseFile, sr=None)
    noise_wav = equalizingRMS(standard_wav,noise_wav) #standard_wav 와 noise_wav 의 RMS 를 같도록
    noise_waves.append(noise_wav) #noise_wav 값을 noise_waves 리스트에 추가
    
# =============================================================================
# Train Data Augmentation (with normalized to standard waveform)
# =============================================================================

noisy_LPSs = [] # noisy 파일들의 LPS feature 들을 저장해두는 리스트
clean_LPSs = [] # clean 파일들의 LPS feature 들을 저장해두는 리스트

for trainFile in [f for f in listdir(PATH_TRAIN) if f.endswith(".wav")]:
    train_wav, rate = librosa.load(PATH_TRAIN + '/' + trainFile, sr=None)
    train_wav = equalizingRMS(standard_wav, train_wav)
    clean_LPS = np.log10(spectrogram(train_wav, power=2))
    for noise in noise_waves:
        for snr in range(-5, 15 + 1, 5):
            noisy_wav = addNoise(train_wav, noise, snr)
            noisy_LPS = np.log10(spectrogram(noisy_wav, power=2))
            noisy_LPSs.append(noisy_LPS) #noisy_LPSs 리스트에 noise_LPS 를 추가한다.
            clean_LPSs.append(clean_LPS) #clean_LPSs 리스트에 clean_LPS 를 추가한다.
            
noisy_LPS_ = np.hstack(noisy_LPSs) #noisy_LPSs 리스트를 가로축으로 순서대로 쌓는다.
clean_LPS_ = np.hstack(clean_LPSs) #clean_LPSs 리스트를 가로축으로 순서대로 쌓는다.

data = {'train_noisy': noisy_LPS_, 'train_clean': clean_LPS_}

# =============================================================================
# Save
# =============================================================================
if not os.path.exists(PATH_FEATURE_OUTPUT):
    os.makedirs(PATH_FEATURE_OUTPUT) #폴더가 존재하지 않을 경우 만듬
    
filename = "%s/Train_feature.pickle" % PATH_FEATURE_OUTPUT
with open(filename, 'wb') as handle:
     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)