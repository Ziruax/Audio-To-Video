import streamlit as st
import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageEnhance
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
import re
from collections import Counter
from transformers import pipeline
import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    CACHE_DIR = Path('cache')
    MAX_CACHE_SIZE = 100 * 1024 * 1024  # 100MB
    MAX_WORKERS = min(os.cpu_count() or 1, 4)
    MAX_IMAGES_PER_KEYWORD = 1
    VIDEO_FPS = 24
    MIN_IMAGE_RESOLUTION = (150, 150)
    MAX_AUDIO_DURATION = 600  # 10 minutes
    DEFAULT_SEGMENT_DURATION = 5
    MAX_KEYWORDS = 3

# Initialize resources
def initialize_resources():
    try:
        Config.CACHE_DIR.mkdir(exist_ok=True)
        logger.info("Cache directory initialized")
        return ThreadPoolExecutor(max_workers=Config.MAX_WORKERS)
    except Exception as e:
        logger.error(f"Failed to initialize resources: {e}")
        raise

thread_pool = initialize_resources()

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
                file.unlink()
        logger.info(f"Cache size: {current_size / 1024 / 1024:.2f} MB")
    except Exception as e:
        logger.warning(f"Cache management error: {e}")

@lru_cache(maxsize=50)
def get_file_hash(file_bytes):
    return hashlib.sha256(file_bytes).hexdigest()

# Initialize transformers pipelines (local, no API)
def initialize_pipelines():
    try:
        asr_pipeline = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h", device=-1)  # CPU
        ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", device=-1)
        logger.info("Transformers pipelines initialized")
        return asr_pipeline, ner_pipeline
    except Exception as e:
        logger.error(f"Failed to initialize pipelines: {e}")
        raise

asr_pipeline, ner_pipeline = initialize_pipelines()

# Audio transcription and keyword extraction
def extract_audio_keywords(audio_file, segment_duration=Config.DEFAULT_SEGMENT_DURATION):
    try:
        file_bytes = audio_file.getvalue()
        file_hash = get_file_hash(file_bytes)
        cache_file = Config.CACHE_DIR / f'segments_{file_hash}.json'

        if cache_file.exists():
            import json
            logger.info(f"Using cached segments: {cache_file}")
            return json.loads(cache_file.read_text())

        audio_data, sample_rate = sf.read(BytesIO(file_bytes))
        total_duration = len(audio_data) / sample_rate
        
        if total_duration > Config.MAX_AUDIO_DURATION:
            raise ValueError(f"Audio exceeds {Config.MAX_AUDIO_DURATION // 60} minutes")

        segments = []
        num_segments = int(total_duration // segment_duration)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, audio_data, sample_rate)
            for i in range(num_segments):
                start = i * segment_duration
                end = min((i + 1) * segment_duration, total_duration)
                segment_audio = audio_data[int(start * sample_rate):int(end * sample_rate)]
                
                # Write segment to temporary file for ASR
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as seg_tmp:
                    sf.write(seg_tmp.name, segment_audio, sample_rate)
                    transcription = asr_pipeline(seg_tmp.name)["text"].lower()
                    os.unlink(seg_tmp.name)
                
                # NER for entities
                ner_results = ner_pipeline(transcription)
                entities = [res["word"] for res in ner_results if res["entity"].startswith("B-") and len(res["word"]) > 3]
                
                # Frequency analysis for non-entities
                words = re.findall(r'\b\w+\b', transcription)
                stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'was', 'are'}
                word_freq = Counter(word for word in words if word not in stop_words and len(word) > 3)
                freq_keywords = [word for word, _ in word_freq.most_common(3)]
                
                # Combine and limit keywords
                keywords = list(dict.fromkeys(entities + freq_keywords))[:Config.MAX_KEYWORDS]
                if not keywords:
                    keywords = ["scene"]  # Fallback, but should be rare with transcription
                
                segments.append({"start": start, "end": end, "keywords": keywords})
                logger.info(f"Segment {i}: Keywords {keywords} from '{transcription}'")
            
            if total_duration % segment_duration:
                start = num_segments * segment_duration
                end = total_duration
                segment_audio = audio_data[int(start * sample_rate):int(end * sample_rate)]
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as seg_tmp:
                    sf.write(seg_tmp.name, segment_audio, sample_rate)
                    transcription = asr_pipeline(seg_tmp.name)["text"].lower()
                    os.unlink(seg_tmp.name)
                
                ner_results = ner_pipeline(transcription)
                entities = [res["word"] for res in ner_results if res["entity"].startswith("B-") and len(res["word"]) > 3]
                word_freq = Counter(word for word in re.findall(r'\b\w+\b', transcription) if word not in stop_words and len(word) > 3)
                freq_keywords = [word for word, _ in word_freq.most_common(3)]
                keywords = list(dict.fromkeys(entities + freq_keywords))[:Config.MAX_KEYWORDS] or ["scene"]
                
                segments.append({"start": start, "end": end, "keywords": keywords})
                logger.info(f"Final segment: Keywords {keywords} from '{transcription}'")

        os.unlink(tmp.name)
        import json
        cache_file.write_text(json.dumps(segments))
        logger.info(f"Segmented audio into {len(segments)} segments with keywords")
        return segments
    except Exception as e:
        logger.error(f"Audio segmentation error: {e}")
        st.error(f"Failed to process audio: {e}")
        return None

# Image processing
def get_resolution(video_format, quality):
    resolutions = {
        "Low": {"9:16": (180, 320), "16:9": (320, 180), "1:1": (240, 240)},
        "Medium": {"9:16": (360, 640), "16:9": (640, 360), "1:1": (480, 480)},
        "High": {"9:16": (720, 1280), "16:9": (1280, 720), "1:1": (960, 960)}
    }
    return resolutions[quality].get(video_format, resolutions[quality]["16:9"])

def validate_image(img, keywords):
    try:
        width, height = img.size
        if width < Config.MIN_IMAGE_RESOLUTION[0] or height < Config.MIN_IMAGE_RESOLUTION[1]:
            logger.warning(f"Image rejected: Size {width}x{height}")
            return False
        img_array = np.array(img)
        brightness = np.mean(img_array)
        if brightness < 10 or brightness > 245:
            logger.warning(f"Image rejected: Brightness {brightness}")
            return False
        return True
    except Exception as e:
        logger.warning(f"Image validation failed: {e}")
        return False

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_image(url):
    try:
        if not validators.url(url):
            raise ValueError("Invalid URL")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert('RGB')
        logger.info(f"Fetched image from {url}")
        return img
    except Exception as e:
        logger.warning(f"Failed to fetch image from {url}: {e}")
        raise

def scrape_images(keywords, num_images=Config.MAX_IMAGES_PER_KEYWORD):
    try:
        cache_key = hashlib.sha256("".join(keywords).encode()).hexdigest()
        cache_dir = Config.CACHE_DIR / f'img_{cache_key}'
        cache_dir.mkdir(exist_ok=True)
        
        cached_images = []
        for p in cache_dir.glob('*.png'):
            try:
                img = Image.open(p)
                if validate_image(img, keywords):
                    cached_images.append(img)
            except Exception as e:
                logger.warning(f"Failed to load cached image {p}: {e}")
        
        if len(cached_images) >= num_images:
            logger.info(f"Using {len(cached_images)} cached images for '{keywords}'")
            return cached_images[:num_images]

        platforms = [
            lambda kw: f'https://www.google.com/search?q={quote(kw)}&tbm=isch',
            lambda kw: f'https://www.bing.com/images/search?q={quote(kw)}',
            lambda kw: f'https://www.pexels.com/search/{quote(kw)}/',
            lambda kw: f'https://unsplash.com/s/photos/{quote(kw)}'
        ]
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        images = []
        for keyword in keywords:
            img_urls = set()
            for platform_fn in random.sample(platforms, len(platforms)):
                platform = platform_fn(keyword)
                try:
                    response = requests.get(platform, headers=headers, timeout=10)
                    response.raise_for_status()
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    if 'google.com' in platform:
                        for img in soup.select('img[src]'):
                            src = img['src']
                            if src.startswith('http') and not src.startswith('data:') and not src.endswith('.gif'):
                                img_urls.add(src)
                    elif 'bing.com' in platform:
                        for img in soup.select('img[src]'):
                            src = img['src']
                            if src.startswith('http') and not src.endswith('.gif'):
                                img_urls.add(src)
                    else:
                        for img in soup.find_all('img'):
                            src = img.get('src')
                            if src and src.startswith('http') and not src.endswith('.gif'):
                                img_urls.add(src)
                    
                    logger.info(f"Found {len(img_urls)} URLs from {platform}")
                except Exception as e:
                    logger.warning(f"Failed to scrape {platform}: {e}")
            
            futures = [thread_pool.submit(fetch_image, url) for url in list(img_urls)[:num_images * 5]]
            for i, future in enumerate(as_completed(futures)):
                try:
                    img = future.result()
                    if validate_image(img, keywords):
                        img_path = cache_dir / f'img_{i}_{keyword}.png'
                        img.save(img_path)
                        images.append(img)
                        logger.info(f"Saved image {img_path}")
                        if len(images) >= num_images:
                            break
                except Exception as e:
                    logger.warning(f"Image fetch failed: {e}")
            
            if images:
                break
        
        if not images:
            refined_keywords = [kw + " photo" for kw in keywords]
            for keyword in refined_keywords:
                for platform_fn in platforms:
                    platform = platform_fn(keyword)
                    try:
                        response = requests.get(platform, headers=headers, timeout=10)
                        soup = BeautifulSoup(response.text, 'html.parser')
                        img_urls = set()
                        if 'google.com' in platform:
                            for img in soup.select('img[src]'):
                                src = img['src']
                                if src.startswith('http') and not src.startswith('data:') and not src.endswith('.gif'):
                                    img_urls.add(src)
                        elif 'bing.com' in platform:
                            for img in soup.select('img[src]'):
                                src = img['src']
                                if src.startswith('http') and not src.endswith('.gif'):
                                    img_urls.add(src)
                        else:
                            for img in soup.find_all('img'):
                                src = img.get('src')
                                if src and src.startswith('http') and not src.endswith('.gif'):
                                    img_urls.add(src)
                        
                        futures = [thread_pool.submit(fetch_image, url) for url in list(img_urls)[:num_images * 5]]
                        for i, future in enumerate(as_completed(futures)):
                            img = future.result()
                            if validate_image(img, keywords):
                                img_path = cache_dir / f'img_{i}_{keyword}.png'
                                img.save(img_path)
                                images.append(img)
                                logger.info(f"Saved refined image {img_path}")
                                if len(images) >= num_images:
                                    break
                        if images:
                            break
                    except Exception as e:
                        logger.warning(f"Retry failed for {platform}: {e}")
                if images:
                    break
        
        if not images:
            raise ValueError(f"Failed to scrape any valid images for '{keywords}' after retries")
        
        return images[:num_images]
    except Exception as e:
        logger.error(f"Image scraping error for '{keywords}': {e}")
        raise

def enhance_image(image, resolution):
    try:
        image = image.resize(resolution, Image.Resampling.LANCZOS)
        enhanced = ImageEnhance.Contrast(image).enhance(1.1)
        logger.info(f"Enhanced image to {resolution}")
        return enhanced
    except Exception as e:
        logger.error(f"Image enhancement error: {e}")
        return image

# Video creation
def create_video(segments, audio_path, video_format, quality):
    output_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
    try:
        audio = AudioFileClip(audio_path)
        resolution = get_resolution(video_format, quality)
        
        clips = []
        for segment in segments:
            if not isinstance(segment, dict) or "keywords" not in segment:
                logger.error(f"Invalid segment format: {segment}")
                raise ValueError(f"Segment must be a dict with 'keywords': {segment}")
            
            keywords = segment["keywords"]
            images = scrape_images(keywords)
            
            img = images[0]
            enhanced_img = enhance_image(img, resolution)
            try:
                clip = ImageSequenceClip([np.array(enhanced_img)], fps=Config.VIDEO_FPS)
                duration = segment["end"] - segment["start"]
                clip = clip.set_duration(duration)
                clip = clip.set_start(segment["start"])
                clip = vfx.fadein(clip, min(0.5, duration/2)).fx(vfx.fadeout, min(0.5, duration/2))
                clips.append(clip)
                logger.info(f"Added clip for {segment['start']}-{segment['end']} with keywords {keywords}")
            except Exception as e:
                logger.error(f"Clip creation failed for segment: {e}")
                continue
        
        if not clips:
            raise ValueError("No valid clips created")
        
        final_clip = concatenate_videoclips(clips, method="compose")
        final_clip = final_clip.set_audio(audio)
        
        preset = {"Low": "veryfast", "Medium": "fast", "High": "medium"}[quality]
        final_clip.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            fps=Config.VIDEO_FPS,
            preset=preset,
            threads=Config.MAX_WORKERS,
            logger=None
        )
        logger.info(f"Video created at {output_path} with {quality} quality")
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
    st.markdown(f"Convert audio to video with relevant images. Max {Config.MAX_AUDIO_DURATION // 60} min.")

    with st.sidebar:
        st.header("Settings")
        num_images = st.slider("Images per keyword", 1, 2, Config.MAX_IMAGES_PER_KEYWORD)
        segment_duration = st.slider("Segment Duration (seconds)", 3, 10, Config.DEFAULT_SEGMENT_DURATION)
        video_format = st.selectbox("Video Format", ["16:9", "9:16", "1:1"], index=0)
        quality = st.selectbox("Video Quality", ["Low", "Medium", "High"], index=1)  # Medium default

    audio_file = st.file_uploader(f"Upload Audio File (Max {Config.MAX_AUDIO_DURATION // 60} min)", type=['wav', 'mp3', 'm4a'])
    
    if audio_file:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Generate Video"):
                with st.spinner("Processing..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Transcribing audio...")
                    segments = extract_audio_keywords(audio_file, segment_duration)
                    if not segments:
                        return
                    progress_bar.progress(0.2)
                    
                    status_text.text("Fetching images...")
                    for i, seg in enumerate(segments):
                        progress_bar.progress(0.2 + (i + 1) * 0.5 / len(segments))
                    
                    status_text.text("Creating video...")
                    with tempfile.NamedTemporaryFile(suffix='.mp3') as tmp_audio:
                        tmp_audio.write(audio_file.getvalue())
                        video_path = create_video(segments, tmp_audio.name, video_format, quality)
                    
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
                    else:
                        st.error("Video creation failed. Check logs.")
                        
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
