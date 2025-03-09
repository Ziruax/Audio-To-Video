import streamlit as st
import librosa
import numpy as np
import moviepy.editor as mpy
from PIL import Image, ImageEnhance
import os
import logging
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import random
import tempfile
import hashlib
import time
import spacy
import soundfile as sf
import whisper

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    CACHE_DIR = Path("cache")
    MAX_CACHE_SIZE = 100 * 1024 * 1024  # 100MB
    MAX_AUDIO_DURATION = 600  # 10 minutes
    DEFAULT_SEGMENT_DURATION = 5
    MAX_KEYWORDS = 3
    VIDEO_FPS = 24
    MIN_IMAGE_RESOLUTION = (150, 150)
    MAX_WORKERS = min(os.cpu_count() or 1, 4)

# Initialize resources
def initialize_resources():
    Config.CACHE_DIR.mkdir(exist_ok=True)
    logger.info("Cache directory initialized")
    return ThreadPoolExecutor(max_workers=Config.MAX_WORKERS)

thread_pool = initialize_resources()

# Load models once
@st.cache_resource
def load_models():
    logger.info("Loading ML models...")
    whisper_model = whisper.load_model("tiny", device="cpu")
    kandinsky_model = None
    try:
        from diffusers import KandinskyV22Pipeline
        kandinsky_model = KandinskyV22Pipeline.from_pretrained("kandinsky-community/kandinsky-2-2-decoder").to("cpu")
        logger.info("Kandinsky model loaded successfully.")
    except Exception as e:
        logger.warning(f"Failed to load Kandinsky model: {e}. AI image generation will be disabled.")
    return whisper_model, kandinsky_model

whisper_model, kandinsky_model = load_models()

# Load spacy model once
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

# Cache management
def manage_cache():
    try:
        current_size = sum(f.stat().st_size for f in Config.CACHE_DIR.rglob('*') if f.is_file())
        if current_size > Config.MAX_CACHE_SIZE:
            cache_files = [(f, f.stat().st_atime) for f in Config.CACHE_DIR.rglob('*') if f.is_file()]
            cache_files.sort(key=lambda x: x[1])
            for file, _ in cache_files:
                if current_size <= Config.MAX_CACHE_SIZE * 0.8:
                    break
                current_size -= file.stat().st_size
                file.unlink()
        logger.info(f"Cache size: {current_size / 1024 / 1024:.2f} MB")
    except Exception as e:
        logger.warning(f"Cache management error: {e}")

# Audio transcription and keyword extraction
def transcribe_segment(args):
    audio_data, sample_rate, start, end = args
    try:
        segment_audio = audio_data[int(start * sample_rate):int(end * sample_rate)]
        # Normalize audio
        segment_audio = segment_audio.astype(np.float32) / (np.max(np.abs(segment_audio)) or 1)
        # Save temporary wav file
        audio_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        sf.write(audio_path, segment_audio, sample_rate)
        result = whisper_model.transcribe(audio_path, language="en")
        os.unlink(audio_path)
        return result["text"]
    except Exception as e:
        logger.warning(f"Transcription error for segment {start}-{end}: {e}")
        return ""

def extract_keywords_from_transcription(transcription):
    try:
        doc = nlp(transcription.lower())
        # Use named entities if available
        entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "GPE", "ORG", "EVENT", "NORP"]]
        if not entities:
            words = [token.text for token in doc if token.is_alpha and not token.is_stop and len(token.text) > 2]
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            entities = [word for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)][:Config.MAX_KEYWORDS]
        return list(dict.fromkeys(entities))[:Config.MAX_KEYWORDS] or ["default"]
    except Exception as e:
        logger.warning(f"Keyword extraction error: {e}")
        return ["default"]

@st.cache_data
def extract_transcribed_keywords(audio_file, segment_duration=Config.DEFAULT_SEGMENT_DURATION):
    try:
        file_bytes = audio_file.getvalue()
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        cache_file = Config.CACHE_DIR / f'segments_{file_hash}.json'

        if cache_file.exists():
            logger.info(f"Using cached segments: {cache_file}")
            return json.loads(cache_file.read_text())

        audio_data, sample_rate = librosa.load(BytesIO(file_bytes), sr=16000)
        total_duration = len(audio_data) / sample_rate
        
        if total_duration > Config.MAX_AUDIO_DURATION:
            raise ValueError(f"Audio exceeds {Config.MAX_AUDIO_DURATION // 60} minutes")

        segments = []
        num_segments = int(total_duration // segment_duration) + (1 if total_duration % segment_duration else 0)
        
        # Create segment tasks
        segment_args = [(audio_data, sample_rate, i * segment_duration, min((i + 1) * segment_duration, total_duration)) 
                        for i in range(num_segments)]
        # Preserve order by collecting results in order
        transcriptions = [thread_pool.submit(transcribe_segment, arg).result() for arg in segment_args]
        
        used_keywords = set()
        for i, transcription in enumerate(transcriptions):
            keywords = extract_keywords_from_transcription(transcription)
            # Filter out duplicate keywords per segment
            filtered_keywords = []
            for kw in keywords:
                if kw not in used_keywords:
                    filtered_keywords.append(kw)
                    used_keywords.add(kw)
            if not filtered_keywords:
                filtered_keywords = ["scene"]
            start = i * segment_duration
            end = min((i + 1) * segment_duration, total_duration)
            segments.append({"start": start, "end": end, "transcription": transcription, "keywords": filtered_keywords})
            logger.info(f"Segment {i}: Transcription '{transcription}' -> Keywords {filtered_keywords}")
        
        cache_file.write_text(json.dumps(segments))
        logger.info(f"Segmented audio into {len(segments)} segments")
        return segments
    except Exception as e:
        logger.error(f"Audio processing error: {e}")
        st.error(f"Failed to process audio: {e}")
        return None

# Image generation
def generate_image_task(transcription, keywords):
    try:
        if not kandinsky_model:
            raise ValueError("Kandinsky model not available")
        prompt = f"A high quality, vibrant image depicting {transcription}, featuring {' and '.join(keywords)}, detailed and colorful"
        image = kandinsky_model(prompt, num_inference_steps=5, height=180, width=320).images[0]
        logger.info(f"Generated image for prompt: {prompt}")
        return image
    except Exception as e:
        logger.error(f"Image generation error: {e}")
        # Fallback: return a plain gray image
        return Image.new("RGB", (320, 180), "gray")

def generate_images(transcription, keywords):
    # Direct call without threading for simplicity
    return generate_image_task(transcription, keywords)

# Image enhancement
def crop_to_aspect(image, aspect_ratio):
    target_aspect = aspect_ratio[0] / aspect_ratio[1]
    width, height = image.size
    current_aspect = width / height
    if current_aspect > target_aspect:
        new_width = int(height * target_aspect)
        left = (width - new_width) // 2
        image = image.crop((left, 0, left + new_width, height))
    else:
        new_height = int(width / target_aspect)
        top = (height - new_height) // 2
        image = image.crop((0, top, width, top + new_height))
    return image

def enhance_image(image, resolution, aspect_ratio):
    try:
        image = crop_to_aspect(image, aspect_ratio)
        image = image.resize(resolution, Image.Resampling.LANCZOS)
        return ImageEnhance.Contrast(image).enhance(1.2)
    except Exception:
        return image

# Video creation
def process_segment(segment, resolution, aspect_ratio):
    img = generate_images(segment["transcription"], segment["keywords"])
    enhanced_img = enhance_image(img, resolution, aspect_ratio)
    clip = mpy.ImageSequenceClip([np.array(enhanced_img)], fps=Config.VIDEO_FPS)
    duration = segment["end"] - segment["start"]
    clip = clip.set_duration(duration).set_start(segment["start"])
    # Optional: Remove or adjust dynamic resize if not needed
    # clip = clip.resize(lambda t: 1 + 0.1 * (t / duration))
    return clip

def create_video(segments, audio_path, quality, video_format):
    output_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
    try:
        audio = mpy.AudioFileClip(audio_path)
        aspect_ratio = {"16:9": (16, 9), "9:16": (9, 16), "1:1": (1, 1)}[video_format]
        resolutions = {
            "Low": {"16:9": (320, 180), "9:16": (180, 320), "1:1": (240, 240)},
            "Medium": {"16:9": (640, 360), "9:16": (360, 640), "1:1": (480, 480)},
            "High": {"16:9": (960, 540), "9:16": (540, 960), "1:1": (720, 720)}
        }
        resolution = resolutions[quality][video_format]
        
        # Process segments in order
        futures = [thread_pool.submit(process_segment, segment, resolution, aspect_ratio) for segment in segments]
        clips = [future.result() for future in futures]
        
        if not clips:
            raise ValueError("No valid clips created")
        
        final_clip = mpy.concatenate_videoclips(clips, method="compose").set_audio(audio)
        preset = {"Low": "veryfast", "Medium": "fast", "High": "medium"}[quality]
        final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=Config.VIDEO_FPS, preset=preset, logger=None)
        logger.info(f"Video created at {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Video creation failed: {str(e)}")
        st.error("Video generation failed. Please check the audio file and try again.")
        return None
    finally:
        if 'final_clip' in locals():
            final_clip.close()
        if 'audio' in locals():
            audio.close()
        for clip in clips:
            clip.close()

# Streamlit UI
def main():
    st.set_page_config(page_title="Audio to Video (AI Generated)", layout="wide")
    st.title("🎥 Audio to Video Generator (AI-Generated Images)")
    st.markdown(f"Convert audio to video using AI-generated images. Max {Config.MAX_AUDIO_DURATION // 60} min.")
    if not kandinsky_model:
        st.error("AI image generation model failed to load. Please check logs and dependencies.")
        return

    with st.sidebar:
        st.header("Settings")
        segment_duration = st.slider("Segment Duration (seconds)", 3, 10, Config.DEFAULT_SEGMENT_DURATION)
        video_format = st.selectbox("Video Format", ["16:9", "9:16", "1:1"], index=0)
        quality = st.selectbox("Video Quality", ["Low", "Medium", "High"], index=1)

    audio_file = st.file_uploader("Upload Audio File (mp3, wav, m4a)", type=["mp3", "wav", "m4a"])
    
    if audio_file:
        st.subheader("Audio Transcription")
        with st.spinner("Transcribing audio..."):
            segments = extract_transcribed_keywords(audio_file, segment_duration)
            if not segments:
                st.error("Transcription failed.")
                return
        
        for i, segment in enumerate(segments):
            st.write(f"Segment {i} ({segment['start']:.1f}s - {segment['end']:.1f}s):")
            st.text(f"Transcription: {segment['transcription']}")
            st.text(f"Keywords: {', '.join(segment['keywords'])}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Generate Video"):
                progress_bar = st.progress(0)
                progress_bar.progress(0.2)
                with st.spinner("Generating video..."):
                    st.info("Processing audio segments...")
                    # Save the uploaded audio to a temporary file
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_audio:
                        tmp_audio.write(audio_file.getvalue())
                        tmp_audio_path = tmp_audio.name
                    progress_bar.progress(0.5)
                    video_path = create_video(segments, tmp_audio_path, quality, video_format)
                    progress_bar.progress(0.8)
                    
                    if video_path:
                        st.success("Video generated successfully!")
                        st.video(video_path)
                        with open(video_path, "rb") as f:
                            st.download_button(
                                "Download Video",
                                f,
                                file_name=f"ai_video_{time.strftime('%Y%m%d_%H%M%S')}.mp4",
                                mime="video/mp4"
                            )
                        os.unlink(video_path)
                        os.unlink(tmp_audio_path)
                    else:
                        st.error("Video generation failed.")
                        progress_bar.progress(0)
        with col2:
            st.audio(audio_file)

if __name__ == "__main__":
    try:
        manage_cache()
        main()
    except Exception as e:
        logger.critical(f"Application error: {e}")
        st.error(f"Unexpected error: {e}")
    finally:
        thread_pool.shutdown(wait=False)
