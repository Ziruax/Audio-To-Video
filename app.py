import streamlit as st
import librosa
import numpy as np
import moviepy.editor as mpy
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from diffusers import StableDiffusionPipeline
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
    MAX_KEYWORDS = 3
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

# Initialize lightweight models
try:
    asr_pipeline = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h", device=-1)
    ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", device=-1)
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    text_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    image_gen_pipeline = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
    image_gen_pipeline = image_gen_pipeline.to("cpu")  # Ensure CPU usage for Streamlit Cloud
    logger.info("All models initialized successfully")
except Exception as e:
    logger.error(f"Model initialization failed: {e}")
    st.error(f"Failed to initialize models: {e}")
    asr_pipeline = ner_pipeline = tokenizer = text_model = image_gen_pipeline = None

# Audio transcription and keyword extraction
def extract_audio_keywords(audio_file, segment_duration=Config.DEFAULT_SEGMENT_DURATION):
    try:
        file_bytes = audio_file.getvalue()
        file_hash = hashlib.sha256(file_bytes).hexdigest()
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
        
        for i in range(num_segments + (1 if total_duration % segment_duration else 0)):
            start = i * segment_duration
            end = min((i + 1) * segment_duration, total_duration)
            segment_audio = audio_data[int(start * sample_rate):int(end * sample_rate)]
            segment_audio = segment_audio.astype(np.float32) / (np.max(np.abs(segment_audio)) or 1)  # Avoid division by zero
            
            # Transcription
            transcription = asr_pipeline({"raw": segment_audio, "sampling_rate": sample_rate})["text"].lower()
            
            # NER for entities
            ner_results = ner_pipeline(transcription)
            entities = [res["word"] for res in ner_results if res["entity"].startswith("B-") and len(res["word"]) > 2]
            
            # Refine with distilgpt2 if needed
            if not entities:
                inputs = tokenizer(transcription, return_tensors="pt", truncation=True, max_length=50)
                outputs = text_model.generate(**inputs, max_new_tokens=20, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
                refined_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                entities = [word for word in refined_text.split() if len(word) > 2][:Config.MAX_KEYWORDS]
            
            keywords = list(dict.fromkeys(entities))[:Config.MAX_KEYWORDS] or ["scene"]
            segments.append({"start": start, "end": end, "keywords": keywords, "transcription": transcription})
            logger.info(f"Segment {i}: Keywords {keywords} from '{transcription}'")
        
        cache_file.write_text(json.dumps(segments))
        logger.info(f"Segmented audio into {len(segments)} segments")
        return segments
    except Exception as e:
        logger.error(f"Audio processing error: {e}")
        st.error(f"Failed to process audio: {e}")
        return None

# Image scraping
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_image(url):
    try:
        if not validators.url(url):
            raise ValueError("Invalid URL")
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert('RGB')
        logger.info(f"Fetched image from {url}")
        return img
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
        
        cached_images = []
        for p in cache_dir.glob('*.png'):
            try:
                img = Image.open(p)
                if img.size >= Config.MIN_IMAGE_RESOLUTION:
                    cached_images.append(img)
            except Exception:
                pass
        
        if cached_images:
            logger.info(f"Using cached image for '{keywords}'")
            return cached_images[0]

        platforms = [
            lambda kw: f'https://www.google.com/search?q={quote(kw)}&tbm=isch',
            lambda kw: f'https://www.bing.com/images/search?q={quote(kw)}',
            lambda kw: f'https://www.pexels.com/search/{quote(kw)}/',
            lambda kw: f'https://unsplash.com/s/photos/{quote(kw)}'
        ]
        
        headers = {'User-Agent': 'Mozilla/5.0'}
        for keyword in keywords:
            if keyword in used_keywords:
                continue
            img_urls = set()
            for platform_fn in platforms:
                platform = platform_fn(keyword)
                try:
                    response = requests.get(platform, headers=headers, timeout=10)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    for img in soup.select('img[src]'):
                        src = img['src']
                        if src.startswith('http') and not src.endswith('.gif'):
                            img_urls.add(src)
                except Exception as e:
                    logger.warning(f"Failed to scrape {platform}: {e}")
            
            futures = [thread_pool.submit(fetch_image, url) for url in list(img_urls)[:5]]
            for future in as_completed(futures):
                try:
                    img = future.result()
                    if img.size >= Config.MIN_IMAGE_RESOLUTION:
                        img_path = cache_dir / f'img_{keyword}.png'
                        img.save(img_path)
                        used_keywords.add(keyword)
                        logger.info(f"Saved scraped image {img_path}")
                        return img
                except Exception:
                    pass
        
        logger.warning(f"No valid images scraped for '{keywords}'")
        return Image.new("RGB", (640, 360), "gray")  # Fallback image
    except Exception as e:
        logger.error(f"Image scraping error: {e}")
        return Image.new("RGB", (640, 360), "gray")

# Image generation
def generate_image(keywords, transcription):
    try:
        prompt = f"{transcription}, high quality, detailed, vibrant colors"
        image = image_gen_pipeline(prompt, num_inference_steps=20).images[0]  # Reduced steps for speed
        logger.info(f"Generated image for prompt: {prompt}")
        return image
    except Exception as e:
        logger.error(f"Image generation error: {e}")
        return Image.new("RGB", (640, 360), "gray")

# Image enhancement
def enhance_image(image, resolution):
    try:
        image = image.resize(resolution, Image.Resampling.LANCZOS)
        return ImageEnhance.Contrast(image).enhance(1.2)
    except Exception:
        return image

# Video creation
def create_video(segments, audio_path, image_source, video_format, quality):
    output_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
    try:
        audio = mpy.AudioFileClip(audio_path)
        resolution = {"Low": (640, 360), "Medium": (960, 540), "High": (1280, 720)}[quality]  # Simplified resolutions
        
        clips = []
        used_keywords = set()
        for segment in segments:
            keywords = tuple(segment["keywords"])
            if keywords in used_keywords:
                keywords = (keywords[0] + str(random.randint(1, 1000)),)
            used_keywords.add(keywords)
            
            if image_source == "scrape":
                img = scrape_images(segment["keywords"], used_keywords)
            else:
                img = generate_image(segment["keywords"], segment["transcription"])
            
            enhanced_img = enhance_image(img, resolution)
            clip = mpy.ImageSequenceClip([np.array(enhanced_img)], fps=Config.VIDEO_FPS)
            clip = clip.set_duration(segment["end"] - segment["start"]).set_start(segment["start"])
            clip = clip.fx(mpy.vfx.fadein, 0.5).fx(mpy.vfx.fadeout, 0.5)
            clips.append(clip)
        
        final_clip = mpy.concatenate_videoclips(clips, method="compose").set_audio(audio)
        preset = {"Low": "veryfast", "Medium": "fast", "High": "medium"}[quality]
        final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=Config.VIDEO_FPS, preset=preset)
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
        video_format = st.selectbox("Video Format", ["16:9"], index=0)  # Simplified for now
        quality = st.selectbox("Video Quality", ["Low", "Medium", "High"], index=1)

    audio_file = st.file_uploader("Upload Audio File (mp3, wav, m4a)", type=["mp3", "wav", "m4a"])
    
    if audio_file:
        image_source = st.radio("Choose Image Source", ["Scrape from Web", "Generate with AI"], index=0)
        image_source_key = "scrape" if image_source == "Scrape from Web" else "generate"
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Generate Video"):
                with st.spinner("Processing audio..."):
                    segments = extract_audio_keywords(audio_file, segment_duration)
                    if not segments:
                        return
                    
                    with st.spinner(f"{'Scraping images' if image_source_key == 'scrape' else 'Generating images'}..."):
                        with tempfile.NamedTemporaryFile(suffix=".mp3") as tmp_audio:
                            tmp_audio.write(audio_file.getvalue())
                            video_path = create_video(segments, tmp_audio.name, image_source_key, video_format, quality)
                    
                    if video_path:
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
