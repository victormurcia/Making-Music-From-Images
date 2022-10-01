# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 11:44:34 2022

@author: vmurc
"""

#Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import cv2
import random
from pedalboard import Pedalboard, Chorus, Reverb, Gain, LadderFilter,Phaser, Delay, PitchShift, Distortion
from pedalboard.io import AudioFile
from PIL import Image
from scipy.io import wavfile
import librosa
import glob

#This function generates frequencies in Hertz from notes
def get_piano_notes():   
    # White keys are in Uppercase and black keys (sharps) are in lowercase
    octave = ['C', 'c', 'D', 'd', 'E', 'F', 'f', 'G', 'g', 'A', 'a', 'B'] 
    base_freq = 440 #Frequency of Note A4
    keys = np.array([x+str(y) for y in range(0,9) for x in octave])
    # Trim to standard 88 keys
    start = np.where(keys == 'A0')[0][0]
    end = np.where(keys == 'C8')[0][0]
    keys = keys[start:end+1]
    
    note_freqs = dict(zip(keys, [2**((n+1-49)/12)*base_freq for n in range(len(keys))]))
    note_freqs[''] = 0.0 # stop
    return note_freqs


#Make scale as specified by user
def makeScale(whichOctave, whichKey, whichScale):
    
    #Load note dictionary
    note_freqs = get_piano_notes()
    
    #Define tones. Upper case are white keys in piano. Lower case are black keys
    scale_intervals = ['A','a','B','C','c','D','d','E','F','f','G','g']
    
    #Find index of desired key
    index = scale_intervals.index(whichKey)
    
    #Redefine scale interval so that scale intervals begins with whichKey
    new_scale = scale_intervals[index:12] + scale_intervals[:index]
    
    #Choose scale
    if whichScale == 'AEOLIAN':
        scale = [0, 2, 3, 5, 7, 8, 10]
    elif whichScale == 'BLUES':
        scale = [0, 2, 3, 4, 5, 7, 9, 10, 11]
    elif whichScale == 'PHYRIGIAN':
        scale = [0, 1, 3, 5, 7, 8, 10]
    elif whichScale == 'CHROMATIC':
        scale = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    elif whichScale == 'DORIAN':
        scale = [0, 2, 3, 5, 7, 9, 10]
    elif whichScale == 'HARMONIC_MINOR':
        scale = [0, 2, 3, 5, 7, 8, 11]
    elif whichScale == 'LYDIAN':
        scale = [0, 2, 4, 6, 7, 9, 11]
    elif whichScale == 'MAJOR':
        scale = [0, 2, 4, 5, 7, 9, 11]
    elif whichScale == 'MELODIC_MINOR':
        scale = [0, 2, 3, 5, 7, 8, 9, 10, 11]
    elif whichScale == 'MINOR':    
        scale = [0, 2, 3, 5, 7, 8, 10]
    elif whichScale == 'MIXOLYDIAN':     
        scale = [0, 2, 4, 5, 7, 9, 10]
    elif whichScale == 'NATURAL_MINOR':   
        scale = [0, 2, 3, 5, 7, 8, 10]
    elif whichScale == 'PENTATONIC':    
        scale = [0, 2, 4, 7, 9]
    else:
        print('Invalid scale name')
    
    #Initialize arrays
    freqs = []
    for i in range(len(scale)):
        note = new_scale[scale[i]] + str(whichOctave)
        freqToAdd = note_freqs[note]
        freqs.append(freqToAdd)
    return freqs

#Convery Hue value to a frequency
def hue2freq(h,scale_freqs):
    thresholds = [26 , 52 , 78 , 104,  128 , 154 , 180]
    #note = scale_freqs[0]
    if (h <= thresholds[0]):
         note = scale_freqs[0]
    elif (h > thresholds[0]) & (h <= thresholds[1]):
        note = scale_freqs[1]
    elif (h > thresholds[1]) & (h <= thresholds[2]):
        note = scale_freqs[2]
    elif (h > thresholds[2]) & (h <= thresholds[3]):
        note = scale_freqs[3]
    elif (h > thresholds[3]) & (h <= thresholds[4]):    
        note = scale_freqs[4]
    elif (h > thresholds[4]) & (h <= thresholds[5]):
        note = scale_freqs[5]
    elif (h > thresholds[5]) & (h <= thresholds[6]):
        note = scale_freqs[6]
    else:
        note = scale_freqs[0]
    
    return note

#Make song from image!
def img2music(img, scale = [220.00, 246.94 ,261.63, 293.66, 329.63, 349.23, 415.30],
              sr = 22050, T = 0.1, nPixels = 60, useOctaves = True, randomPixels = False,
              harmonize = 'U0'):
    """
    Args:
        img    :     (array) image to process
        scale  :     (array) array containing frequencies to map H values to
        sr     :     (int) sample rate to use for resulting song
        T      :     (int) time in seconds for dutation of each note in song
        nPixels:     (int) how many pixels to use to make song
    Returns:
        song   :     (array) Numpy array of frequencies. Can be played by ipd.Audio(song, rate = sr)
    """
    #Convert image to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    #Get shape of image
    height, width, depth = img.shape

    i=0 ; j=0 ; k=0
    #Initialize array the will contain Hues for every pixel in image
    hues = [] 
    if randomPixels == False:
        for val in range(nPixels):
                hue = abs(hsv[i][j][0]) #This is the hue value at pixel coordinate (i,j)
                hues.append(hue)
                i+=1
                j+=1
    else:
        for val in range(nPixels):
            i = random.randint(0, height-1)
            j = random.randint(0, width-1)
            hue = abs(hsv[i][j][0]) #This is the hue value at pixel coordinate (i,j)
            hues.append(hue)
             
    #Make dataframe containing hues and frequencies
    pixels_df = pd.DataFrame(hues, columns=['hues'])
    pixels_df['frequencies'] = pixels_df.apply(lambda row : hue2freq(row['hues'],scale), axis = 1) 
    frequencies = pixels_df['frequencies'].to_numpy()
    
    #Convert frequency to a note
    pixels_df['notes'] = pixels_df.apply(lambda row : librosa.hz_to_note(row['frequencies']), axis = 1)  
    
    #Convert note to a midi number
    pixels_df['midi_number'] = pixels_df.apply(lambda row : librosa.note_to_midi(row['notes']), axis = 1)  
    
    #Make harmony dictionary
    #unison           = U0 ; semitone         = ST ; major second     = M2
    #minor third      = m3 ; major third      = M3 ; perfect fourth   = P4
    #diatonic tritone = DT ; perfect fifth    = P5 ; minor sixth      = m6
    #major sixth      = M6 ; minor seventh    = m7 ; major seventh    = M7
    #octave           = O8
    harmony_select = {'U0' : 1,
                      'ST' : 16/15,
                      'M2' : 9/8,
                      'm3' : 6/5,
                      'M3' : 5/4,
                      'P4' : 4/3,
                      'DT' : 45/32,
                      'P5' : 3/2,
                      'm6': 8/5,
                      'M6': 5/3,
                      'm7': 9/5,
                      'M7': 15/8,
                      'O8': 2
                     }
    harmony = np.array([]) #This array will contain the song harmony
    harmony_val = harmony_select[harmonize] #This will select the ratio for the desired harmony
                                               
    #song_freqs = np.array([]) #This array will contain the chosen frequencies used in our song :]
    song = np.array([])       #This array will contain the song signal
    octaves = np.array([0.5,1,2])#Go an octave below, same note, or go an octave above
    t = np.linspace(0, T, int(T*sr), endpoint=False) # time variable
    #Make a song with numpy array :]
    #nPixels = int(len(frequencies))#All pixels in image
    for k in range(nPixels):
        if useOctaves:
            octave = random.choice(octaves)
        else:
            octave = 1
        
        if randomPixels == False:
            val =  octave * frequencies[k]
        else:
            val = octave * random.choice(frequencies)
            
        #Make note and harmony note    
        note   = 0.5*np.sin(2*np.pi*val*t)
        h_note = 0.5*np.sin(2*np.pi*harmony_val*val*t)  
        
        #Place notes into corresponfing arrays
        song       = np.concatenate([song, note])
        harmony    = np.concatenate([harmony, h_note])                                     
        #song_freqs = np.concatenate([song_freqs, val])
                                               
    return song, pixels_df, harmony


# Adding an appropriate title for the test website
st.title("Making Music From Images")

st.markdown("This little app converts an image into a song. Play around with the various inputs belows using different images!")
#Making dropdown select box containing scale, key, and octave choices
df1 = pd.DataFrame({'Scale_Choice': ['AEOLIAN', 'BLUES', 'PHYRIGIAN', 'CHROMATIC','DORIAN','HARMONIC_MINOR','LYDIAN','MAJOR','MELODIC_MINOR','MINOR','MIXOLYDIAN','NATURAL_MINOR','PENTATONIC']})
df2 = pd.DataFrame({'Keys': ['A','a','B','C','c','D','d','E','F','f','G','g']})
df3 = pd.DataFrame({'Octaves': [1,2,3]})
df4 = pd.DataFrame({'Harmonies': ['U0','ST','M2','m3','M3','P4','DT','P5','m6','M6','m7','M7','O8']})

st.sidebar.markdown("Select sample image if you'd like to use one of the preloaded images. Select User Image is you'd like to use your own image.")
_radio = st.sidebar.radio("",("Use Sample Image", "Use User Image"))

sample_images = glob.glob('*.jpg')
samp_imgs_df = pd.DataFrame(sample_images,columns=['Images'])
samp_img = st.sidebar.selectbox('Choose a sample image', samp_imgs_df['Images'])

#Load image 
user_data = st.sidebar.file_uploader(label="Upload your own Image")
if _radio == "Use Sample Image":
    img2load = samp_img
elif _radio == "Use User Image": 
    img2load = user_data

#Display the image
st.sidebar.image(img2load)    

col1, col2, col3, col4 = st.columns(4)

with col1:
    scale = st.selectbox('What scale would you like yo use?', df1['Scale_Choice'])

    'You selected the ' + scale + ' scale'
with col2:
    key = st.selectbox('What key would you like to use?', df2['Keys']) 
    
    'You selected: ', key

with col3:
    octave = st.selectbox('What octave would you like to use?', df3['Octaves']) 

    'You selected: ', octave
with col4:
    harmony = st.selectbox('What harmony would you like to use?', df4['Harmonies']) 

    'You selected: ', harmony

col5, col6 = st.columns(2)
with col5:
    #Ask user if they want to use random pixels
    random_pixels = st.checkbox('Use random pixels to build song?', value=True)
with col6:
    #Ask user to select song duration
    use_octaves = st.checkbox('Randomize note octaves while building song?', value=True) 
    
col7, col8 = st.columns(2)
with col7:
    #Ask user to select song duration
    t_value = st.slider('Note duration [s]', min_value=0.01, max_value=1.0, step = 0.01, value=0.2)     

with col8:
    #Ask user to select song duration
    n_pixels = st.slider('How many pixels to use? (More pixels take longer)', min_value=12, max_value=320, step=1, value=60)         
#***Start Peadalboard Definitions*** 
st.markdown("## Pedalboard")
col9, col10,col11,col12 = st.columns(4)
#Chorus Parameters
with col9:
    st.markdown("### Chorus Parameters")
    rate_hz_chorus = st.slider('rate_hz', min_value=0.0, max_value=100.0, step=0.1, value=0.0)  

#Delay Parameters
with col10:
    st.markdown("### Delay Parameters")
    delay_seconds = st.slider('delay_seconds', min_value=0.0, max_value=2.0, step=0.1, value=0.0)  
    
#Distortion Parameters
with col11:
    st.markdown("### Distortion Parameters")
    drive_db = st.slider('drive_db', min_value=0.0, max_value=100.0, step=1.0, value=0.0) 

#Gain Parameters
with col12:
    st.markdown("### Gain Parameters")
    gain_db = st.slider('gain_db', min_value=0.0, max_value=100.0, step=1.0, value=0.0) 

st.markdown("### Reverb Parameters")
rev1, rev2, rev3, rev4, rev5= st.columns(5)
#Reverb Parameters
with rev1:
    room_size = st.slider('room_size', min_value=0.0, max_value=1.0, step=0.1, value=0.0) 
with rev2:
    damping   = st.slider('damping'  , min_value=0.0, max_value=1.0, step=0.1, value=0.0) 
with rev3:
    wet_level = st.slider('wet_level', min_value=0.0, max_value=1.0, step=0.1, value=0.0) 
with rev4:
    dry_level = st.slider('dry_level', min_value=0.1, max_value=1.0, step=0.1, value=0.1) 
with rev5:
    width     = st.slider('width'    , min_value=0.0, max_value=1.0, step=0.1, value=0.0)

st.markdown("### Ladder Filter Parameters")
lf1,lf2,lf3 = st.columns(3)
#Ladder Filter Parameters
with lf1:
    cutoff_hz     = st.slider('cutoff_hz', min_value=0.0, max_value=1000.0, step=1.0, value=0.0) 
with lf2:
    resonance_lad = st.slider('resonance', min_value=0.0, max_value=1.0, step=0.1, value=0.0)
with lf3:
    drive_lad     = st.slider('drive', min_value=1.0, max_value=100.0, step=0.1, value=1.0)

#st.markdown("### Phaser Parameters")
ch1,ps1 = st.columns(2) 
#Phaser Parameters
with ch1:
    st.markdown("### Phaser Parameters")
    rate_hz_phaser = st.slider('rate_hz_phaser', min_value=0.0, max_value=100.0, step=0.1, value=0.0)  
    depth_phaser   = st.slider('depth', min_value=0.0, max_value=1.0, step=0.1, value=0.0) 

with ps1:
    st.markdown("### Pitch Shift Parameters")
    semitones   = st.slider('semitones', min_value=0.0, max_value=12.0, step=1.0, value=0.0) 

# Making the required prediction
if img2load is not None:
    # Saves
    img = Image.open(img2load)
    img = img.save("img.jpg")
    
    # OpenCv Read
    img = cv2.imread("img.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    #Display the image
    #st.image(img)
    
    #Make the scale from parameters above
    scale_to_use = makeScale(octave, key, scale)
    
    #Make the song!
    song, song_df,harmony = img2music(img, scale = scale_to_use, T = t_value, randomPixels = random_pixels, useOctaves = use_octaves,nPixels = n_pixels,harmonize = harmony)
    
    #Write the song into a file
    song_combined = np.vstack((song, harmony))
    wavfile.write('song.wav', rate = 22050, data = song_combined.T.astype(np.float32))
    
    audio_file = open('song.wav', 'rb')
    audio_bytes = audio_file.read()
    
    # Read in a whole audio file:
    with AudioFile('song.wav', 'r') as f:
        audio = f.read(f.frames)
        samplerate = f.samplerate
        
    # Make a Pedalboard object, containing multiple plugins:
    board = Pedalboard([
        Gain(gain_db=gain_db),
        Distortion(drive_db=drive_db),
        LadderFilter(mode=LadderFilter.Mode.HPF12, cutoff_hz=cutoff_hz,resonance = resonance_lad,drive=drive_lad),
        Delay(delay_seconds = delay_seconds),
        Reverb(room_size = room_size, wet_level = wet_level, dry_level = dry_level, width = width),
        Phaser(rate_hz = rate_hz_phaser, depth = depth_phaser),
        PitchShift(semitones = semitones),
        Chorus(rate_hz = rate_hz_chorus)
        ])

    # Run the audio through this pedalboard!
    effected = board(audio, samplerate)
    
    # Write the audio back as a wav file:
    with AudioFile('processed_song.wav', 'w', samplerate, effected.shape[0]) as f:
        f.write(effected)
        
    #Read the processed song
    audio_file2 = open('processed_song.wav', 'rb')
    audio_bytes2 = audio_file2.read()   
    
    #Play the processed song
    st.audio(audio_bytes2, format='audio/wav')
    
    #@st.cache
    def convert_df_to_csv(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')
    #csv = song_df.to_csv('song.csv')
    st.download_button('Download Song as CSV', data=convert_df_to_csv(song_df), file_name="song.csv",mime='text/csv',key='download-csv')
 # While no image is uploaded
else:
    st.write("Waiting for an image to be uploaded...")
#st.markdown("# Main page 🎈")
#st.sidebar.markdown("# Main page 🎈")