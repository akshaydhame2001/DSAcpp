## GStreamer

GStreamer is an open-source, cross-platform multimedia framework used to build complex media-handling pipelines.
It allows you to process audio, video, and streaming data in real time or offline by combining modular plugins into pipelines.

### Benefits of GStreamer

- Video encoding and decoding
- Efficient streaming or handling of multiple video streams
- Supported Streaming Protocols: UDP, RTSP, HTTP, WebRTC
- Supported Formats: H.264, H.265, MJPEG, MP4, AVI, MKV
- Real-time processing
- Cross-platform compatibility: Linux, Windows, MacOS, Android, and embedded platforms

Basic pipeline structure:

```
gst-launch-1.0 [source] ! [decoder] ! [processing] ! [sink]
```

### Common Jetson Elements

- `nvarguscamerasrc`: Access CSI camera
- `nvvidconv`: Hardware-accelerated color conversion and scaling
- `nvv4l2decoder`: Hardware-accelerated video decoding
- `nvv4l2h264enc` / `nvv4l2h265enc`: Hardware-accelerated video encoding
- `nvjpegenc` / `nvjpegdec`: JPEG encoding and decoding
- `nvegltransform`: Transformation for EGL rendering
- `nveglglessink`: Efficient rendering to screen

### CSI Camera Pipeline Example

```
gst-launch-1.0 nvarguscamerasrc ! \
'video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1' ! \
nvvidconv ! nvegltransform ! nveglglessink -e
```

### USB Camera Pipeline (OpenCV example)

```python
import cv2

gst_str = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw, width=640, height=480, framerate=30/1 ! "
    "videoconvert ! appsink"
)

cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
```

### RTSP Stream and Decoder Example

```
gst-launch-1.0 rtspsrc location=rtsp://IP:PORT ! \
rtph264depay ! h264parse ! nvv4l2decoder ! \
nvvidconv ! nvegltransform ! nveglglessink
```

### Useful Commands

- Check supported camera formats:

```
v4l2-ctl --list-formats-ext
```

- Show GStreamer version:

```
gst-inspect-1.0 --version
```

- List all GStreamer plugins:

```
gst-inspect-1.0
```

- Debug pipeline issues (increase debug level as needed):

```
GST_DEBUG=3 gst-launch-1.0 [pipeline]
```
