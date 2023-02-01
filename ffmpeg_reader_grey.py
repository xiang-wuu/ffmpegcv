import numpy as np
import warnings
from .video_info import (run_async, get_info, get_num_NVIDIA_GPUs, 
                        decoder_to_nvidia)
from .ffmpeg_reader import FFmpegReaderNV


class FFmpegReaderGreyNV(FFmpegReaderNV):
    @staticmethod
    def VideoReader(filename, pix_fmt, crop_xywh, 
                    resize, resize_keepratio, resize_keepratioalign, 
                    gpu):
        if pix_fmt is not None:
            warnings.warn('The `pix_fmt` is not used in FFmpegReaderGreyNV!')
        numGPU = get_num_NVIDIA_GPUs()
        assert numGPU>0, 'No GPU found'
        gpu = int(gpu) % numGPU if gpu is not None else 0
        assert resize is None or len(resize) == 2, 'resize must be a tuple of (width, height)'
        videoinfo = get_info(filename)
        vid = FFmpegReaderGreyNV()
        cropopt, scaleopt, filteropt = vid._get_opts(videoinfo, crop_xywh, resize, 
            resize_keepratio, resize_keepratioalign)
        vid.codecNV = decoder_to_nvidia(vid.codec)
        
        args = (f'ffmpeg -loglevel warning -hwaccel cuda -hwaccel_device {gpu} '
                f' -vcodec {vid.codecNV} {cropopt} {scaleopt} -r {vid.fps} -i "{filename}" '
                f' {filteropt} -pix_fmt nv12 -r {vid.fps} -f rawvideo pipe:')

        vid.process = run_async(args)
        return vid

    def read(self):
        in_bytes = self.process.stdout.read(self.height * self.width * 3 // 2)
        if not in_bytes:
            self.release()
            return False, None
        img_btyes = np.frombuffer(in_bytes, dtype=np.uint8)
        img_grey = img_btyes[:self.height*self.width].reshape([self.height, self.width, 1])
        return True, img_grey
