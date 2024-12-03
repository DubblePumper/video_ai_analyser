import moviepy.editor as mp
import speech_recognition as sr
import numpy as np
import librosa
import pydub
from pydub import AudioSegment
import tempfile
import os
import glob

def extract_audio_from_video(video_path):
    """
    Extract audio from the video using moviepy and save it as a .wav file.
    """
    try:
        # Load the video file
        video = mp.VideoFileClip(video_path)

        # Extract audio from the video
        audio = video.audio

        # Temporary file to save the audio
        audio_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        audio.write_audiofile(audio_file.name)
        
        return audio_file.name
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None

def transcribe_audio(audio_file_path):
    """
    Transcribe speech from an audio file using SpeechRecognition library.
    """
    try:
        recognizer = sr.Recognizer()

        # Open the audio file and recognize the speech
        with sr.AudioFile(audio_file_path) as source:
            audio_data = recognizer.record(source)
        
        # Recognize speech using Google's speech recognition API
        text = recognizer.recognize_google(audio_data)
        return text
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None

def analyze_audio_features(audio_file_path):
    """
    Analyze audio features like volume, pitch, etc. using librosa.
    """
    try:
        # Load the audio file
        audio_data, sr = librosa.load(audio_file_path)

        # Extract audio features
        pitch, mag = librosa.core.piptrack(y=audio_data, sr=sr)
        volume = np.mean(np.abs(audio_data))  # RMS volume

        # Return the extracted features
        features = {
            "pitch": pitch,
            "volume": volume
        }
        return features
    except Exception as e:
        print(f"Error analyzing audio: {e}")
        return None

def analyze_audio_sentiment(text):
    """
    (Optional) Analyze the sentiment of the transcribed text.
    This can be done using a sentiment analysis library like TextBlob or VADER.
    """
    from textblob import TextBlob
    try:
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        return sentiment
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return None

def analyze_video_audio(video_path):
    """
    Main function to extract, transcribe, and analyze audio from a video.
    """
    audio_file = extract_audio_from_video(video_path)
    
    if audio_file:
        # Step 1: Transcribe the audio
        transcription = transcribe_audio(audio_file)

        # Step 2: Analyze audio features like pitch and volume
        audio_features = analyze_audio_features(audio_file)

        # Step 3: (Optional) Analyze sentiment of the transcription
        sentiment = None
        if transcription:
            sentiment = analyze_audio_sentiment(transcription)

        # Return all results
        results = {
            "transcription": transcription,
            "audio_features": audio_features,
            "sentiment": sentiment
        }

        # Clean up the temporary audio file
        os.remove(audio_file)

        return results
    else:
        print("Error: Audio extraction failed.")
        return None

def process_all_video_files_in_folder(folder_path):
    """
    Automatically processes all video files in the given folder.
    """
    # Get all video files in the folder (you can adjust the extensions as needed)
    video_files = glob.glob(os.path.join(folder_path, "*.mp4")) + \
                  glob.glob(os.path.join(folder_path, "*.avi")) + \
                  glob.glob(os.path.join(folder_path, "*.mov"))  # Add more formats if needed

    results = {}

    # Process each video file found in the folder
    for video_path in video_files:
        print(f"Processing video: {video_path}")
        video_results = analyze_video_audio(video_path)
        
        if video_results:
            results[video_path] = video_results

    return results

# Example usage: Automatically process all video files in the 'input_videos/' folder
folder_path = 'app/input_videos'  # Replace with your folder path if different
all_video_analysis_results = process_all_video_files_in_folder(folder_path)

# Print results for each video
for video, analysis in all_video_analysis_results.items():
    print(f"Results for {video}:")
    print(f"Transcription: {analysis['transcription']}")
    print(f"Audio Features: {analysis['audio_features']}")
    print(f"Sentiment: {analysis['sentiment']}")