## FFMPEGCV is an alternative to OPENCV for video read and write.

This is just an effort to maintain the code base of official [ffmpegcv](https://pypi.org/project/ffmpegcv/) python module, as the main author hasn't exposed the code base, this is just to make the code available for more contribution's and make it accessible to the open source community.

The ffmpegcv provide Video Reader and Video Witer with ffmpeg backbone, which are faster and powerful than cv2.

- The ffmpegcv is api compatible to open-cv.
- The ffmpegcv can use GPU accelerate encoding and decoding.
- The ffmpegcv support much more video codecs v.s. open-cv.
- The ffmpegcv support RGB & BGR format as you like.
- The ffmpegcv can resize video to specific size with/without padding.

In all, ffmpegcv is just similar to opencv api. But is faster and with more codecs.

### Basic example

Read a video by GPU, and rewrite it.

```
vidin = ffmpegcv.VideoCaptureNV(vfile_in)
vidout = ffmpegcv.VideoWriter(vfile_out, 'h264', vidin.fps)

with vidin, vidout:
    for frame in vidin:
        cv2.imshow('image', frame)
        vidout.write(frame)
```

### Install

You need to download ffmpeg before you can use ffmpegcv

```bash
conda install ffmpeg

pip install ffmpegcv
```

### GPU Accelation

- Support NVIDIA card only.
- Perfect in the Windows. That ffmpeg supports NVIDIA acceleration just by conda install.
- Struggle in the Linux. That ffmpeg didn't orginally support NVIDIA accelerate. Please re-compile the ffmpeg by yourself. See the [link](https://docs.nvidia.com/video-technologies/video-codec-sdk/ffmpeg-with-nvidia-gpu/)
- Infeasible in the MacOS. That ffmpeg didn't supports NVIDIA at all.

### Video Reader
<hr/>
The ffmpegcv is just similar to opencv in api.

```
# open cv
import cv2
cap = cv2.VideoCapture(file)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    pass

# ffmpegcv
import ffmpegcv
cap = ffmpegcv.VideoCapture(file)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    pass
cap.release()

# alternative
cap = ffmpegcv.VideoCapture(file)
nframe = len(cap)
for frame in cap:
    pass
cap.release()

# more pythonic, recommand
with ffmpegcv.VideoCapture(file) as cap:
    nframe = len(cap)
    for iframe, frame in enumerate(cap):
        if iframe>100: break
        pass
```

Use GPU to accelerate decoding. It depends on the video codes. h264_nvcuvid, hevc_nvcuvid ....

```
cap_cpu = ffmpegcv.VideoCapture(file)
cap_gpu = ffmpegcv.VideoCapture(file, codec='h264_cuvid') #NVIDIA GPU0
cap_gpu0 = ffmpegcv.VideoCaptureNV(file)         #NVIDIA GPU0
cap_gpu1 = ffmpegcv.VideoCaptureNV(file, gpu=1)  #NVIDIA GPU1
```

Use rgb24 instead of bgr24

```
cap = ffmpegcv.VideoCapture(file, pix_fmt='rgb24')
ret, frame = cap.read()
plt.imshow(frame)
```

Crop video, which will be much faster than read the whole canvas.

```
cap = ffmpegcv.VideoCapture(file, crop_xywh=(0, 0, 640, 480))
```

Resize the video to the given size.

```
cap = ffmpegcv.VideoCapture(file, resize=(640, 480))
```

Resize and keep the aspect ratio with black border padding.

```
cap = ffmpegcv.VideoCapture(file, resize=(640, 480), resize_keepratio=True)
```

Crop and then resize the video.

```
cap = ffmpegcv.VideoCapture(file, crop_xywh=(0, 0, 640, 480), resize=(512, 512))
```

## Contact

For any professional/commercial support kindly reach out to [xiangwuu4@gmail.com](mailto:xiangwuu4@gmail.com).

Many Thanks to original author [chenxf](mailto:cxf529125853@163.com).

