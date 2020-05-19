import ffmpeg_streaming
from ffmpeg_streaming import Formats, Representation, Size, Bitrate

if __name__ == "__main__":
    video = ffmpeg_streaming.input('output.mp4')
    hls = video.hls(Formats.h264())
    hls.auto_generate_representations()
    hls.output('static/hls.m3u8')