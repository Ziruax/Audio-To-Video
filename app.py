import streamlit as st
import torch
from transformers import pipeline
import spacy
import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont
import cv2
import numpy as np
from moviepy.editor import ImageSequenceClip, VideoFileClip, AudioFileClip, concatenate_videoclips, vfx
import os
import soundfile as sf
from io import BytesIO
import time
import logging
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import hashlib
from pathlib import Path
import random
from tenacity import retry, stop_after_attempt, wait_exponential
import tempfile
import validators
import subprocess

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    CACHE_DIR = Path('cache')
    MAX_CACHE_SIZE = 500 * 1024 * 1024  # 500MB
    MAX_WORKERS = min(os.cpu_count() * 2, 10) if os.cpu_count() else 4
    MAX_IMAGES_PER_KEYWORD = 5
    VIDEO_FPS = 24
    MIN_IMAGE_RESOLUTION = (400, 400)
    MAX_AUDIO_DURATION = 3000  # 50 minutes
    DEFAULT_SEGMENT_DURATION = 5  # seconds

# Initialize resources with robust model downloading
def initialize_resources():
    Config.CACHE_DIR.mkdir(exist_ok=True)
    
    # Ensure Spacy model is downloaded
    model_name = 'en_core_web_sm'
    try:
        nlp = spacy.load(model_name)
    except OSError:
        logger.info(f"Spacy model '{model_name}' not found. Attempting to download...")
        try:
            # Use Spacy's built-in download command with a check for success
            result = subprocess.run(
                ["python", "-m", "spacy", "download", model_name],
                check=True,
                capture_output=True,
                text=True
            )
            logger.info(f"Spacy model download output: {result.stdout}")
            nlp = spacy.load(model_name)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download Spacy model: {e.stderr}")
            raise RuntimeError(f"Could not download Spacy model '{model_name}': {e.stderr}")
        except Exception as e:
            logger.error(f"Unexpected error downloading Spacy model: {str(e)}")
            raise RuntimeError(f"Unexpected error downloading Spacy model: {str(e)}")
    
    # Initialize other resources
    transcriber = pipeline(
        'automatic-speech-recognition',
        model='openai/whisper-large-v3',
        device=0 if torch.cuda.is_available() else -1
    )
    thread_pool = ThreadPoolExecutor(max_workers=Config.MAX_WORKERS)
    
    return nlp, transcriber, thread_pool

# Initialize resources at startup
try:
    nlp, transcriber, thread_pool = initialize_resources()
except Exception as e:
    st.error(f"Failed to initialize resources: {str(e)}")
    st.stop()

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
                size = file.stat().st_size
                file.unlink()
                current_size -= size
    except Exception as e:
        logger.warning(f"Cache management error: {e}")

@lru_cache(maxsize=100)
def get_file_hash(file_bytes):
    return hashlib.sha256(file_bytes).hexdigest()

# Audio processing with segmentation
def transcribe_and_segment_audio(audio_file, segment_duration=Config.DEFAULT_SEGMENT_DURATION):
    try:
        file_bytes = audio_file.getvalue()
        file_hash = get_file_hash(file_bytes)
        cache_file = Config.CACHE_DIR / f'trans_{file_hash}.json'

        if cache_file.exists():
            import json
            return json.loads(cache_file.read_text())

        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        result = transcriber(tmp_path, return_timestamps=True)
        chunks = result['chunks']
        audio = AudioFileClip(tmp_path)
        total_duration = audio.duration
        
        if total_duration > Config.MAX_AUDIO_DURATION:
            raise ValueError("Audio exceeds 50 minutes")

        num_segments = int(total_duration // segment_duration)
        segmentsycat = []
        
        for i in range(num_segments):
            start = i * segment_duration
            end = min((i + 1) * segment_duration, total_duration)
            segment_chunks = [c for c in chunks if start <= c['timestamp'][0] < end]
            segment_text = " ".join(c['text'] for c in segment_chunks)
            segments.append({"start": start, "end": end, "text": segment_text})
        
        if total_duration % segment_duration:
            start = num_segments * segment_duration
            end = total_duration
            segment_chunks = [c for c in chunks if start <= c['timestamp'][0] < end]
            segment_text = " ".join(c['text'] for c in segment_chunks)
            segments.append({"start": start, "end": end, "text": segment_text})

        import json
        cache_file.write_text(json.dumps(segments))
        
        os.unlink(tmp_path)
        audio.close()
        
        return segments
    except Exception as e:
        logger.error(f"Audio processing error: {e}")
        st.error(f"Failed to process audio: {e}")
        return None

# Text processing
def extract_keywords(text, min_keywords=3, max_keywords=10):
    doc = nlp(text)
    keywords = set()
    
    for ent in doc.ents:
        if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT', 'WORK_OF_ART']:
            keywords.add(ent.text)
    
    if len(keywords) < min_keywords:
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN'] and len(token.text) > 2:
                keywords.add(token.text)
                if len(keywords) >= max_keywords:
                    break
    
    return list(keywords)[:max_keywords]

# Image processing
def get_resolution(video_format):
    return {"9:16": (720, 1280), "16:9": (1920, 1080), "1:1": (1080, 1080)}.get(video_format, (1920, 1080))

def validate_image(img):
    try:
        width, height = img.size
        if width < Config.MIN_IMAGE_RESOLUTION[0] or height < Config.MIN_IMAGE_RESOLUTION[1]:
            return False
        
        img_array = np.array(img)
        brightness = np.mean(img_array)
        if brightness < 30 or brightness > 225:
            return False
            
        sharpness = cv2.Laplacian(cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
        if sharpness < 150:
            return False
            
        return True
    except Exception:
        return False

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_image(url):
    if not validators.url(url):
        raise ValueError("Invalid URL")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers, timeout=15)
    return Image.open(BytesIO(response.content)).convert('RGB')

def scrape_images(keyword, num_images=Config.MAX_IMAGES_PER_KEYWORD):
    cache_key = hashlib.sha256(keyword.encode()).hexdigest()
    cache_dir = Config.CACHE_DIR / f'img_{cache_key}'
    cache_dir.mkdir(exist_ok=True)
    
    cached_images = [Image.open(p) for p in cache_dir.glob('*.png') if validate_image(Image.open(p))]
    if len(cached_images) >= num_images:
        return cached_images[:num_images]

    platforms = [
        f'https://www.google.com/search?q={quote(keyword)}&tbm=isch',
        f'https://unsplash.com/s/photos/{quote(keyword)}',
        f'https://www.pexels.com/search/{quote(keyword)}/'
    ]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    img_urls = set()
    try:
        for platform in random.sample(platforms, min(2, len(platforms))):
            try:
                response = requests.get(platform, headers=headers, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                
                for img in soup.find_all('img'):
                    src = img.get('src')
                    if src and src.startswith('http') and not src.endswith('.gif'):
                        img_urls.add(src)
                        
            except Exception as e:
                logger.warning(f"Failed to scrape {platform}: {e}")
                continue
        
        futures = [thread_pool.submit(fetch_image, url) for url in list(img_urls)[:num_images * 2]]
        images = []
        
        for i, future in enumerate(as_completed(futures)):
            try:
                img = future.result()
                if validate_image(img):
                    img_path = cache_dir / f'img_{i}.png'
                    img.save(img_path)
                    images.append(img)
                    if len(images) >= num_images:
                        break
            except Exception as e:
                logger.warning(f"Image fetch failed: {e}")
                continue
                
        return images if images else cached_images
    except Exception as e:
        logger.error(f"Image scraping error: {e}")
        return cached_images if cached_images else []

def create_text_image(text, resolution):
    img = Image.new('RGB', resolution, color=(73, 109, 137))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        font = ImageFont.load_default()
    d.text((10, 10), text, font=font, fill=(255, 255, 0))
    path = Config.CACHE_DIR / f"text_{int(time.time())}.png"
    img.save(path)
    return path

def enhance_image(image, resolution):
    try:
        image = image.resize(resolution, Image.LANCZOS)
        
        img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        img_array = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)
        lab = cv2.cvtColor(img_array, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        img_array = cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2BGR)
        
        img_pil = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
        img_pil = img_pil.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        enhancer = ImageEnhance.Color(img_pil).enhance(1.15)
        return ImageEnhance.Contrast(enhancer).enhance(1.1)
    except Exception as e:
        logger.error(f"Image enhancement error: {e}")
        return image

# Video creation
def create_video(segments, audio_path, video_format):
    output_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        
    try:
        audio = AudioFileClip(audio_path)
        resolution = get_resolution(video_format)
        
        clips = []
        for segment in segments:
            keywords = extract_keywords(segment['text']) if segment['text'] else ['generic scene']
            images = scrape_images(keywords[0])
            
            if not images:
                img_path = create_text_image(segment['text'] or "No transcription available", resolution)
                img = Image.open(img_path)
            else:
                img = images[0]
            
            enhanced_img = enhance_image(img, resolution)
            clip = ImageSequenceClip([np.array(enhanced_img)], fps=Config.VIDEO_FPS)
            clip = clip.set_duration(segment['end'] - segment['start'])
            clip = clip.set_start(segment['start'])
            clip = vfx.fadein(clip, 0.5).fx(vfx.fadeout, 0.5)
            clip = clip.fx(vfx.resize, lambda t: 1 + 0.02 * t)
            clips.append(clip)
        
        final_clip = concatenate_videoclips(clips, method="compose")
        final_clip = final_clip.set_audio(audio)
        
        final_clip.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            fps=Config.VIDEO_FPS,
            preset='medium',
            threads=Config.MAX_WORKERS,
            logger=None
        )
        return output_path
    except Exception as e:
        logger.error(f"Video creation error: {e}")
        st.error(f"Failed to create video: {e}")
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
    st.set_page_config(page_title="Audio to Video Generator", layout="wide")
    st.title("🎥 Audio to Video Generator")
    st.markdown("Convert your audio into engaging videos with AI-generated visuals")

    with st.sidebar:
        st.header("Settings")
        num_images = st.slider("Images per keyword", 1, 10, Config.MAX_IMAGES_PER_KEYWORD)
        segment_duration = st.slider("Segment Duration (seconds)", 3, 15, Config.DEFAULT_SEGMENT_DURATION)
        video_format = st.selectbox("Video Format", ["16:9", "9:16", "1:1"], index=0)
        quality = st.selectbox("Video Quality", ["Low", "Medium", "High"], index=1)

    audio_file = st.file_uploader("Upload Audio File (Max 50 min)", type=['wav', 'mp3', 'm4a'])
    
    if audio_file:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Generate Video"):
                with st.spinner("Processing..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Processing audio and segmenting...")
                    segments = transcribe_and_segment_audio(audio_file, segment_duration)
                    if not segments:
                        return
                    progress_bar.progress(0.2)
                    
                    status_text.text("Extracting keywords...")
                    for i, seg in enumerate(segments):
                        seg['keywords'] = extract_keywords(seg['text'])
                        progress_bar.progress(0.2 + (i + 1) * 0.3 / len(segments))
                    
                    status_text.text("Creating video...")
                    with tempfile.NamedTemporaryFile(suffix='.mp3') as tmp_audio:
                        tmp_audio.write(audio_file.getvalue())
                        video_path = create_video(segments, tmp_audio.name, video_format)
                    
                    if video_path:
                        progress_bar.progress(1.0)
                        status_text.text("Complete!")
                        st.video(video_path)
                        
                        with open(video_path, 'rb') as f:
                            st.download_button(
                                "Download Video",
                                f,
                                file_name=f"generated_video_{time.strftime('%Y%m%d_%H%M%S')}.mp4",
                                mime="video/mp4"
                            )
                        os.unlink(video_path)
                        
        with col2:
            st.audio(audio_file)

if __name__ == "__main__":
    try:
        manage_cache()
        main()
    except Exception as e:
        logger.critical(f"Application error: {e}")
        st.error("An unexpected error occurred. Please try again.")
    finally:
        thread_pool.shutdown()
