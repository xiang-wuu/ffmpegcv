import numpy as np
import warnings
from .video_info import (run_async, get_info, get_num_NVIDIA_GPUs, 
                        decoder_to_nvidia)
from .ffmpeg_reader import FFmpegReaderNV


class FFmpegReaderLaggy(FFmpegReaderNV):
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
        vid = FFmpegReaderLaggy()
        cropopt, scaleopt, filteropt = vid._get_opts(videoinfo, crop_xywh, resize, 
            resize_keepratio, resize_keepratioalign)
        vid.codecNV = decoder_to_nvidia(vid.codec)
        
        args = (f'ffmpeg -loglevel warning -hwaccel cuda -hwaccel_device {gpu} '
                f' -vcodec {vid.codecNV} {cropopt} {scaleopt} -r {vid.fps} -i "{filename}" '
                f' {filteropt} -pix_fmt nv12 -r {vid.fps} -f rawvideo pipe:')

        vid.process = run_async(args)
        vid._iframe = 0
        vid._framecontainer = []
        assert vid.count>20, 'The video is too short!'
        return vid

    def read(self):
        if self._iframe == 0:
            _, imgnow = self._read()
            self._framecontainer.extend([imgnow]*4)
            for _ in range(3):
                _, imgnext = self._read()
                self._framecontainer.append(imgnext)
        elif self._iframe == self.count:
            self.release()
            return False, None
        else:
            ret, imgnext = self._read()
            imgnext = imgnext if ret else self._framecontainer[-1]
            self._framecontainer.append(imgnext)
            del self._framecontainer[0]
            assert len(self._framecontainer) == 7

        img = np.concatenate([self._framecontainer[0], 
                            self._framecontainer[3], 
                            self._framecontainer[6]], axis=2)
        self._iframe += 1
        return True, img

    def _read(self):
        in_bytes = self.process.stdout.read(self.height * self.width * 3 // 2)
        if not in_bytes:
            return False, None
        img_btyes = np.frombuffer(in_bytes, dtype=np.uint8)
        img_grey = img_btyes[:self.height*self.width].reshape([self.height, self.width, 1])
        return True, img_grey
