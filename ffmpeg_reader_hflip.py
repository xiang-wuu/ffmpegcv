import numpy as np
import warnings
from .video_info import (run_async, get_info, get_num_NVIDIA_GPUs, 
                        decoder_to_nvidia)
from .ffmpeg_reader import FFmpegReaderNV


class FFmpegReaderHFlip(FFmpegReaderNV):
    @staticmethod
    def VideoReader(filename, pix_fmt='bgr24', crop_xywh=None, 
                    resize=None, resize_keepratio=True, resize_keepratioalign='center', 
                    hflip=False, gpu=0):
        numGPU = get_num_NVIDIA_GPUs()
        assert numGPU>0, 'No GPU found'
        gpu = int(gpu) % numGPU if gpu is not None else 0
        assert resize is None or len(resize) == 2, 'resize must be a tuple of (width, height)'
        videoinfo = get_info(filename)
        vid = FFmpegReaderHFlip()
        cropopt, scaleopt, filteropt = vid._get_opts(videoinfo, crop_xywh, resize, 
            resize_keepratio, resize_keepratioalign)
        vid.codecNV = decoder_to_nvidia(vid.codec)
        
        if hflip:
            filteropt = filteropt + ',hflip' if filteropt else '-vf hflip'
        
        args = (f'ffmpeg -loglevel warning -hwaccel cuda -hwaccel_device {gpu} '
                f' -vcodec {vid.codecNV} {cropopt} {scaleopt} -r {vid.fps} -i "{filename}" '
                f' {filteropt} -pix_fmt {pix_fmt} -r {vid.fps} -f rawvideo pipe:')

        vid.process = run_async(args)
        vid._iframe = 0
        vid._framecontainer = []
        assert vid.count>20, 'The video is too short!'
        return vid
