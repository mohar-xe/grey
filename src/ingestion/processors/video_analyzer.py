from urllib.parse import urlparse
from core.exceptions import ProcessingError
from youtube_transcript_api import YouTubeTranscriptApi
import re

def _is_youtube_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in ('http', 'https'):
        return False
    if parsed.netloc in ('www.youtube.com', 'youtube.com', 'm.youtube.com'):
        return parsed.path.startswith(('/watch', '/embed', '/shorts'))
    if parsed.netloc in ('youtu.be', 'www.youtu.be'):
        return True
    return False

def _get_video_id(url):
    patterns = [
        r'(?:youtube\.com/shorts/)([a-zA-Z0-9_-]+)',
        r'(?:youtube\.com/watch\?v=)([a-zA-Z0-9_-]+)',
        r'(?:youtu\.be/)([a-zA-Z0-9_-]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def _get_raw_transcirpt(url: str) -> str:
    if _is_youtube_url(url):
        video_id = _get_video_id(url)
    else:
        raise ProcessingError("Not a valid YouTube URL.")
    
    try:
        raw_transcript = YouTubeTranscriptApi().fetch(video_id).to_raw_data()
        return raw_transcript
    except Exception as e:
        raise ProcessingError(f"Failed to fetch transcript: {str(e)}")
    
def format_transcript_by_time(url, chunk_duration_sec=60):

    raw_transcript = _get_raw_transcirpt(url)
    
    chunks = []
    current_chunk_text = []
    current_chunk_start = 0.0
    
    for segment in raw_transcript:
        text = segment['text'].replace('\n', ' ')
        text = re.sub(r'\[.*?\]', '', text).strip()
        
        if not text:
            continue
            
        if not current_chunk_text:
            current_chunk_start = segment['start']
            
        current_chunk_text.append(text)
        
        time_elapsed = (segment['start'] + segment['duration']) - current_chunk_start
        
        if time_elapsed >= chunk_duration_sec:
            chunks.append({
                "start_time": round(current_chunk_start, 2),
                "text": " ".join(current_chunk_text)
            })
            current_chunk_text = []
            
    if current_chunk_text:
        chunks.append({
            "start_time": round(current_chunk_start, 2),
            "text": " ".join(current_chunk_text)
        })
        
    return chunks