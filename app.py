import streamlit as st
import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageEnhance
import numpy as np
import moviepy.editor as mpy  # Import moviepy safely
import os
import librosa
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
import json

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

# Audio analysis and keyword extraction with librosa
def extract_audio_keywords(audio_file, segment_duration=Config.DEFAULT_SEGMENT_DURATION):
    try:
        file_bytes = audio_file.getvalue()
        file_hash = get_file_hash(file_bytes)
        cache_file = Config.CACHE_DIR / f'segments_{file_hash}.json'

        if cache_file.exists():
            logger.info(f"Using cached segments: {cache_file}")
            return json.loads(cache_file.read_text())

        audio_data, sample_rate = librosa.load(BytesIO(file_bytes), sr=None)
        total_duration = len(audio_data) / sample_rate
        
        if total_duration > Config.MAX_AUDIO_DURATION:
            raise ValueError(f"Audio exceeds {Config.MAX_AUDIO_DURATION // 60} minutes")

        segments = []
        num_segments = int(total_duration // segment_duration)
        
        keyword_pool = {
            "high_energy": ["party", "dance", "concert", "festival", "energy"],
            "medium_energy": ["conversation", "meeting", "city", "street", "people"],
            "low_energy": ["nature", "forest", "river", "calm", "sky"],
            "music": ["music", "instrument", "song", "melody", "guitar"],
            "silence": ["background", "abstract", "texture", "pattern", "scene"]
        }

        for i in range(num_segments):
            start = i * segment_duration
            end = min((i + 1) * segment_duration, total_duration)
            segment_audio = audio_data[int(start * sample_rate):int(end * sample_rate)]
            
            energy = np.mean(librosa.feature.rms(y=segment_audio)[0])
            if energy > 0.05:
                category = "high_energy" if energy > 0.1 else "music"
            elif energy > 0.01:
                category = "medium_energy"
            else:
                category = "low_energy" if energy > 0.005 else "silence"
            
            available_keywords = [kw for kw in keyword_pool[category] if not any(kw in seg["keywords"] for seg in segments)]
            if not available_keywords:
                available_keywords = keyword_pool[category]
            keywords = random.sample(available_keywords, min(Config.MAX_KEYWORDS, len(available_keywords)))
            
            segments.append({"start": start, "end": end, "keywords": keywords})
            logger.info(f"Segment {i}: Keywords {keywords} (energy: {energy:.4f})")
        
        if total_duration % segment_duration:
            start = num_segments * segment_duration
            end = total_duration
            segment_audio = audio_data[int(start * sample_rate):int(end * sample_rate)]
            energy = np.mean(librosa.feature.rms(y=segment_audio)[0])
            category = "high_energy" if energy > 0.1 else "medium_energy" if energy > 0.01 else "low_energy" if energy > 0.005 else "silence"
            available_keywords = [kw for kw in keyword_pool[category] if not any(kw in seg["keywords"] for seg in segments)]
            if not available_keywords:
                available_keywords = keyword_pool[category]
            keywords = random.sample(available_keywords, min(Config.MAX_KEYWORDS, len(available_keywords)))
            
            segments.append({"start": start, "end": end, "keywords": keywords})
            logger.info(f"Final segment: Keywords {keywords} (energy: {energy:.4f})")

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

def scrape_images(keywords, num_images=Config.MAX_IMAGES_PER_KEYWORD, used_keywords=None):
    if used_keywords is None:
        used_keywords = set()
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
            if keyword in used_keywords:
                continue
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
                        used_keywords.add(keyword)
                        logger.info(f"Saved image {img_path}")
                        if len(images) >= num_images:
                            break
                except Exception as e:
                    logger.warning(f"Image fetch failed: {e}")
            
            if images:
                break
        
        if not images:
            logger.warning(f"No images found for '{keywords}'")
            raise ValueError(f"Failed to scrape images for '{keywords}'")
        
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
        audio = mpy.AudioFileClip(audio_path)
        resolution = get_resolution(video_format, quality)
        
        clips = []
        used_keywords = set()
        for segment in segments:
            keywords = segment["keywords"]
            images = scrape_images(keywords, used_keywords=used_keywords)
            
            img = images[0]
            enhanced_img = enhance_image(img, resolution)
            clip = mpy.ImageSequenceClip([np.array(enhanced_img)], fps=Config.VIDEO_FPS)
            duration = segment["end"] - segment["start"]
            clip = clip.set_duration(duration).set_start(segment["start"])
            clip = clip.fx(mpy.vfx.fadein, min(0.5, duration/2)).fx(mpy.vfx.fadeout, min(0.5, duration/2))
            clips.append(clip)
            logger.info(f"Added clip for {segment['start']}-{segment['end']} with keywords {keywords}")
        
        if not clips:
            raise ValueError("No valid clips created")
        
        final_clip = mpy.concatenate_videoclips(clips, method="compose").set_audio(audio)
        
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
        logger.info(f"Video created at {output_path}")
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
        segment_duration = st.slider("Segment Duration (seconds)", 3, 10, Config.DEFAULT_SEGMENT_DURATION)
        video_format = st.selectbox("Video Format", ["16:9", "9:16", "1:1"], index=0)
        quality = st.selectbox("Video Quality", ["Low", "Medium", "High"], index=1)

    audio_file = st.file_uploader(f"Upload Audio File (Max {Config.MAX_AUDIO_DURATION // 60} min)", type=['wav', 'mp3', 'm4a'])
    
    if audio_file:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Generate Video"):
                with st.spinner("Processing..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Analyzing audio...")
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
