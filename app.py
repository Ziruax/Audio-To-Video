import streamlit as st
import librosa
import numpy as np
import moviepy.editor as mpy
from diffusers import KandinskyV22Pipeline
import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageEnhance
import os
import logging
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import random
import tempfile
import hashlib
from tenacity import retry, stop_after_attempt, wait_exponential
import validators
import time
import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    CACHE_DIR = Path("cache")
    MAX_CACHE_SIZE = 100 * 1024 * 1024  # 100MB
    MAX_AUDIO_DURATION = 600  # 10 minutes
    DEFAULT_SEGMENT_DURATION = 5
    MAX_KEYWORDS = 2
    VIDEO_FPS = 24
    MIN_IMAGE_RESOLUTION = (150, 150)
    MAX_WORKERS = min(os.cpu_count() or 1, 4)

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

# Audio feature extraction with parallel processing
def analyze_segment(args):
    audio_data, sample_rate, start, end = args
    try:
        segment_audio = audio_data[int(start * sample_rate):int(end * sample_rate)]
        energy = np.mean(librosa.feature.rms(y=segment_audio)[0])
        pitch = np.mean(librosa.pitch_tuning(segment_audio)) if energy > 0.01 else 0
        tempo = librosa.beat.tempo(y=segment_audio, sr=sample_rate)[0] if energy > 0.01 else 0
        return energy, pitch, tempo
    except Exception as e:
        logger.warning(f"Segment analysis error: {e}")
        return 0, 0, 0

def extract_audio_keywords(audio_file, segment_duration=Config.DEFAULT_SEGMENT_DURATION):
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
        
        keyword_pool = {
            "high_energy": ["party", "dance", "concert", "festival", "action", "race"],
            "music": ["music", "song", "melody", "guitar", "piano", "drums"],
            "speech": ["conversation", "meeting", "people", "speech", "discussion", "talk"],
            "calm": ["nature", "forest", "river", "ocean", "sky", "sunset"],
            "silence": ["abstract", "pattern", "texture", "minimal", "background", "scene"]
        }

        # Parallel audio analysis
        segment_args = [(audio_data, sample_rate, i * segment_duration, min((i + 1) * segment_duration, total_duration)) 
                        for i in range(num_segments)]
        futures = [thread_pool.submit(analyze_segment, arg) for arg in segment_args]
        
        for i, future in enumerate(as_completed(futures)):
            energy, pitch, tempo = future.result()
            if energy > 0.1 and tempo > 120:
                category = "high_energy"
            elif energy > 0.05 and tempo > 60:
                category = "music"
            elif energy > 0.02 and abs(pitch) > 0.1:
                category = "speech"
            elif energy > 0.005:
                category = "calm"
            else:
                category = "silence"
            
            keywords = random.sample(keyword_pool[category], min(Config.MAX_KEYWORDS, len(keyword_pool[category])))
            start = i * segment_duration
            end = min((i + 1) * segment_duration, total_duration)
            segments.append({"start": start, "end": end, "keywords": keywords})
            logger.info(f"Segment {i}: Keywords {keywords} (energy: {energy:.4f}, tempo: {tempo:.1f})")
        
        cache_file.write_text(json.dumps(segments))
        logger.info(f"Segmented audio into {len(segments)} segments")
        return segments
    except Exception as e:
        logger.error(f"Audio processing error: {e}")
        st.error(f"Failed to process audio: {e}")
        return None

# Image scraping with parallel processing
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
def fetch_image(url):
    try:
        if not validators.url(url):
            raise ValueError("Invalid URL")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert('RGB')
        if img.size[0] >= Config.MIN_IMAGE_RESOLUTION[0] and img.size[1] >= Config.MIN_IMAGE_RESOLUTION[1]:
            brightness = np.mean(np.array(img))
            if 20 < brightness < 235:  # Avoid too dark or too bright
                return img
        raise ValueError("Image quality invalid")
    except Exception as e:
        logger.warning(f"Failed to fetch image from {url}: {e}")
        raise

def scrape_images(keywords, used_keywords=None):
    if used_keywords is None:
        used_keywords = set()
    try:
        cache_key = hashlib.sha256("".join(keywords).encode()).hexdigest()
        cache_dir = Config.CACHE_DIR / f'img_{cache_key}'
        cache_dir.mkdir(exist_ok=True)
        
        for p in cache_dir.glob('*.png'):
            try:
                img = Image.open(p)
                if img.size >= Config.MIN_IMAGE_RESOLUTION:
                    logger.info(f"Using cached image for '{keywords}'")
                    return img
            except Exception:
                pass

        platforms = [
            f"https://www.google.com/search?q={'+'.join(keywords)}+high+quality&tbm=isch",
            f"https://www.bing.com/images/search?q={'+'.join(keywords)}+photo",
            f"https://www.pexels.com/search/{'+'.join(keywords)}/",
            f"https://unsplash.com/s/photos/{'-'.join(keywords)}"
        ]
        
        headers = {'User-Agent': 'Mozilla/5.0'}
        img_urls = set()
        for platform in platforms:
            try:
                response = requests.get(platform, headers=headers, timeout=5)
                soup = BeautifulSoup(response.text, 'html.parser')
                for img in soup.select('img[src]'):
                    src = img['src']
                    if src.startswith('http') and not src.endswith('.gif') and 'logo' not in src.lower():
                        img_urls.add(src)
            except Exception as e:
                logger.warning(f"Failed to scrape {platform}: {e}")
        
        if not img_urls:
            raise ValueError("No image URLs found")
        
        futures = [thread_pool.submit(fetch_image, url) for url in list(img_urls)[:10]]  # Limit to 10 for speed
        for future in as_completed(futures):
            try:
                img = future.result()
                img_path = cache_dir / f'img_{keywords[0]}_{int(time.time())}.png'
                img.save(img_path)
                used_keywords.update(keywords)
                logger.info(f"Saved scraped image {img_path}")
                return img
            except Exception:
                pass
        
        logger.warning(f"No valid images scraped for '{keywords}'")
        return Image.new("RGB", (320, 180), "gray")
    except Exception as e:
        logger.error(f"Image scraping error for '{keywords}': {e}")
        return Image.new("RGB", (320, 180), "gray")

# Image generation with parallel processing
def generate_image_task(keywords):
    try:
        pipe = KandinskyV22Pipeline.from_pretrained("kandinsky-community/kandinsky-2-2-decoder")
        pipe = pipe.to("cpu")
        prompt = f"{', '.join(keywords)}, high quality, vibrant colors, simple composition"
        image = pipe(prompt, num_inference_steps=10, height=180, width=320).images[0]
        logger.info(f"Generated image for prompt: {prompt}")
        return image
    except Exception as e:
        logger.error(f"Image generation error: {e}")
        return Image.new("RGB", (320, 180), "gray")

def generate_images(keywords):
    future = thread_pool.submit(generate_image_task, keywords)
    return future.result()

# Image enhancement
def enhance_image(image, resolution):
    try:
        image = image.resize(resolution, Image.Resampling.LANCZOS)
        return ImageEnhance.Contrast(image).enhance(1.2)
    except Exception:
        return image

# Video creation with parallel image processing
def process_segment(segment, image_source, resolution, used_keywords):
    keywords = tuple(segment["keywords"])
    if keywords in used_keywords:
        keywords = (keywords[0] + str(random.randint(1, 1000)),)
    used_keywords.add(keywords)
    
    img = scrape_images(segment["keywords"], used_keywords) if image_source == "scrape" else generate_images(segment["keywords"])
    enhanced_img = enhance_image(img, resolution)
    clip = mpy.ImageSequenceClip([np.array(enhanced_img)], fps=Config.VIDEO_FPS)
    clip = clip.set_duration(segment["end"] - segment["start"]).set_start(segment["start"])
    return clip

def create_video(segments, audio_path, image_source, quality):
    output_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
    try:
        audio = mpy.AudioFileClip(audio_path)
        resolution = {"Low": (320, 180), "Medium": (640, 360), "High": (960, 540)}[quality]
        
        used_keywords = set()
        futures = [thread_pool.submit(process_segment, segment, image_source, resolution, used_keywords) 
                   for segment in segments]
        
        clips = []
        for future in as_completed(futures):
            try:
                clip = future.result()
                clips.append(clip)
            except Exception as e:
                logger.warning(f"Clip processing failed: {e}")
        
        if not clips:
            raise ValueError("No valid clips created")
        
        final_clip = mpy.concatenate_videoclips(clips, method="compose").set_audio(audio)
        preset = {"Low": "veryfast", "Medium": "fast", "High": "medium"}[quality]
        final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=Config.VIDEO_FPS, preset=preset, logger=None)
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
    st.markdown(f"Convert audio to video with scraped or generated images. Max {Config.MAX_AUDIO_DURATION // 60} min.")

    with st.sidebar:
        st.header("Settings")
        segment_duration = st.slider("Segment Duration (seconds)", 3, 10, Config.DEFAULT_SEGMENT_DURATION)
        quality = st.selectbox("Video Quality", ["Low", "Medium", "High"], index=1)

    audio_file = st.file_uploader("Upload Audio File (mp3, wav, m4a)", type=["mp3", "wav", "m4a"])
    
    if audio_file:
        image_source = st.radio("Choose Image Source", ["Scrape from Web", "Generate with AI"], index=0)
        image_source_key = "scrape" if image_source == "Scrape from Web" else "generate"
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Generate Video"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Analyzing audio...")
                segments = extract_audio_keywords(audio_file, segment_duration)
                if not segments:
                    return
                progress_bar.progress(0.2)
                
                status_text.text(f"{'Scraping images' if image_source_key == 'scrape' else 'Generating images'}...")
                with tempfile.NamedTemporaryFile(suffix=".mp3") as tmp_audio:
                    tmp_audio.write(audio_file.getvalue())
                    video_path = create_video(segments, tmp_audio.name, image_source_key, quality)
                progress_bar.progress(0.8)
                
                if video_path:
                    status_text.text("Finalizing video...")
                    progress_bar.progress(1.0)
                    st.success("Video generated successfully!")
                    st.video(video_path)
                    with open(video_path, "rb") as f:
                        st.download_button(
                            "Download Video",
                            f,
                            file_name=f"generated_video_{time.strftime('%Y%m%d_%H%M%S')}.mp4",
                            mime="video/mp4"
                        )
                    os.unlink(video_path)
                else:
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
