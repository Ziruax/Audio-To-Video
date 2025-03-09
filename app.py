import streamlit as st
import librosa
import numpy as np
import moviepy.editor as mpy
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
import spacy

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

# Audio transcription and keyword extraction
def transcribe_segment(args):
    audio_data, sample_rate, start, end = args
    try:
        import whisper
        model = whisper.load_model("tiny", device="cpu")
        segment_audio = audio_data[int(start * sample_rate):int(end * sample_rate)]
        segment_audio = segment_audio.astype(np.float32) / (np.max(np.abs(segment_audio)) or 1)
        audio_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        librosa.output.write_wav(audio_path, segment_audio, sample_rate)
        result = model.transcribe(audio_path, language="en")
        os.unlink(audio_path)
        return result["text"]
    except Exception as e:
        logger.warning(f"Transcription error for segment {start}-{end}: {e}")
        return ""

def extract_keywords_from_transcription(transcription):
    try:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(transcription.lower())
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
        
        segment_args = [(audio_data, sample_rate, i * segment_duration, min((i + 1) * segment_duration, total_duration)) 
                        for i in range(num_segments)]
        futures = [thread_pool.submit(transcribe_segment, arg) for arg in segment_args]
        
        used_keywords = set()
        for i, future in enumerate(as_completed(futures)):
            transcription = future.result()
            keywords = extract_keywords_from_transcription(transcription)
            for kw in keywords[:]:  # Avoid duplicates across segments
                if kw in used_keywords:
                    keywords.remove(kw)
            if not keywords:
                keywords = ["scene"]
            used_keywords.update(keywords)
            start = i * segment_duration
            end = min((i + 1) * segment_duration, total_duration)
            segments.append({"start": start, "end": end, "transcription": transcription, "keywords": keywords})
            logger.info(f"Segment {i}: Transcription '{transcription}' -> Keywords {keywords}")
        
        cache_file.write_text(json.dumps(segments))
        logger.info(f"Segmented audio into {len(segments)} segments")
        return segments
    except Exception as e:
        logger.error(f"Audio processing error: {e}")
        st.error(f"Failed to process audio: {e}")
        return None

# Image scraping
def scrape_platform(platform, query):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(platform.format(query=query), headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        img_urls = []
        if 'pexels.com' in platform:
            img_urls = [img['src'].replace('w=500', 'w=1000') for img in soup.select('img.photo-item__img') if 'images.pexels.com' in img['src']]
        elif 'unsplash.com' in platform:
            img_urls = [img['src'].split('?')[0] + '?w=960' for img in soup.select('img.yWaYX') if 'images.unsplash.com' in img['src']]
        elif 'google.com' in platform:
            img_urls = [img['data-src'] for img in soup.select('img[data-src]') if img['data-src'].startswith('http')]
        elif 'bing.com' in platform:
            img_urls = [img['src'] for img in soup.select('img.mimg') if img['src'].startswith('http')]
        return img_urls[:5]
    except Exception as e:
        logger.warning(f"Failed to scrape {platform}: {e}")
        return []

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
def fetch_image(url):
    try:
        if not validators.url(url):
            raise ValueError("Invalid URL")
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert('RGB')
        if img.size[0] >= Config.MIN_IMAGE_RESOLUTION[0] and img.size[1] >= Config.MIN_IMAGE_RESOLUTION[1]:
            brightness = np.mean(np.array(img))
            if 20 < brightness < 235:
                return img
        raise ValueError("Image quality invalid")
    except Exception as e:
        logger.warning(f"Failed to fetch image from {url}: {e}")
        raise

def scrape_images(transcription, keywords, used_keywords=None):
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

        query = f"{transcription} high quality photo {' '.join(keywords)}"
        platforms = [
            "https://www.pexels.com/search/{query}/",
            "https://unsplash.com/s/photos/{query}",
            "https://www.google.com/search?q={query}&tbm=isch",
            "https://www.bing.com/images/search?q={query}",
        ]
        
        futures = [thread_pool.submit(scrape_platform, platform, query) for platform in platforms]
        img_urls = set()
        for future in as_completed(futures):
            img_urls.update(future.result())
        
        if not img_urls:
            raise ValueError("No image URLs found")
        
        futures = [thread_pool.submit(fetch_image, url) for url in list(img_urls)]
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
        return Image.new("RGB", (320, 180), "gray"))

# Image generation
def generate_image_task(transcription, keywords):
    import torch
    from diffusers import KandinskyV22Pipeline
    from PIL import Image
    try:
        pipe = KandinskyV22Pipeline.from_pretrained("kandinsky-community/kandinsky-2-2-decoder")
        pipe = pipe.to("cpu")
        prompt = f"A high quality, vibrant image depicting {transcription}, featuring {' and '.join(keywords)}, detailed and colorful"
        image = pipe(prompt, num_inference_steps=5, height=180, width=320).images[0]
        logger.info(f"Generated image for prompt: {prompt}")
        return image
    except Exception as e:
        logger.error(f"Image generation error: {e}")
        return Image.new("RGB", (320, 180), "gray")

def generate_images(transcription, keywords):
    future = thread_pool.submit(generate_image_task, transcription, keywords)
    return future.result()

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
def process_segment(segment, image_source, resolution, aspect_ratio, used_keywords):
    keywords = tuple(segment["keywords"])
    if keywords in used_keywords:
        keywords = (keywords[0] + str(random.randint(1, 1000)),)
    used_keywords.add(keywords)
    
    img = (scrape_images(segment["transcription"], segment["keywords"], used_keywords) if image_source == "scrape" 
           else generate_images(segment["transcription"], segment["keywords"]))
    enhanced_img = enhance_image(img, resolution, aspect_ratio)
    clip = mpy.ImageSequenceClip([np.array(enhanced_img)], fps=Config.VIDEO_FPS)
    duration = segment["end"] - segment["start"]
    clip = clip.set_duration(duration).set_start(segment["start"])
    clip = clip.resize(lambda t: 1 + 0.1 * (t / duration))
    return clip

def create_video(segments, audio_path, image_source, quality, video_format):
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
        
        used_keywords = set()
        futures = [thread_pool.submit(process_segment, segment, image_source, resolution, aspect_ratio, used_keywords) 
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
        video_format = st.selectbox("Video Format", ["16:9", "9:16", "1:1"], index=0)
        quality = st.selectbox("Video Quality", ["Low", "Medium", "High"], index=1)

    audio_file = st.file_uploader("Upload Audio File (mp3, wav, m4a)", type=["mp3", "wav", "m4a"])
    
    if audio_file:
        st.subheader("Audio Transcription")
        with st.spinner("Transcribing audio..."):
            segments = extract_transcribed_keywords(audio_file, segment_duration)
            if not segments:
                return
        
        # Display transcriptions
        for i, segment in enumerate(segments):
            st.write(f"Segment {i} ({segment['start']:.1f}s - {segment['end']:.1f}s):")
            st.text(f"Transcription: {segment['transcription']}")
            st.text(f"Keywords: {', '.join(segment['keywords'])}")
        
        image_source = st.radio("Choose Image Source", ["Scrape from Web", "Generate with AI"], index=0)
        image_source_key = "scrape" if image_source == "Scrape from Web" else "generate"
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Generate Video"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Processing audio segments...")
                progress_bar.progress(0.2)
                
                status_text.text(f"{'Scraping images' if image_source_key == 'scrape' else 'Generating images'}...")
                with tempfile.NamedTemporaryFile(suffix=".mp3") as tmp_audio:
                    tmp_audio.write(audio_file.getvalue())
                    video_path = create_video(segments, tmp_audio.name, image_source_key, quality, video_format)
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
                    status_text.text("Failed to generate video. Check logs for details.")
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
