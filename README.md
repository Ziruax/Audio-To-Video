# Audio to Video

A Streamlit application that converts audio files into videos with synchronized images.

## Features
- Audio transcription using Whisper-large-v3
- Image scraping from web sources
- Video generation with multiple format options (16:9, 9:16, 1:1)
- Configurable segment duration
- Downloadable video output

## Deployment on Streamlit Cloud
1. Push this repository to GitHub
2. Connect to Streamlit Cloud
3. Set the app file to `app.py`
4. Add `setup.sh` as the "Run" command in Advanced Settings
5. Deploy!

## Requirements
See `requirements.txt` for full list of dependencies.

## Usage
1. Upload an audio file (max 50 minutes)
2. Adjust settings (images per keyword, segment duration, video format, quality)
3. Click "Generate Video"
4. Download the resulting video
