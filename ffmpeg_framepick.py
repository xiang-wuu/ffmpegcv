import numpy as np
from .video_info import run_async, get_info, release_process
import time

class FFmpegFramePick(object):

    @staticmethod
    def VideoFramePick(filename, codec, pix_fmt, crop_xywh,
                    resize, resize_keepratio, resize_keepratioalign):
        
        assert pix_fmt in ['rgb24', 'bgr24']

        vid = FFmpegFramePick()
        videoinfo = get_info(filename)
        vid.filename = filename
        vid.pix_fmt = pix_fmt
        vid.width = videoinfo.width
        vid.height = videoinfo.height
        vid.fps = videoinfo.fps
        vid.count = videoinfo.count
        vid.origin_width, vid.origin_height = vid.width, vid.height
        vid.codec = codec if codec else videoinfo.codec

        if crop_xywh:
            crop_w, crop_h = crop_xywh[2:]
            vid.width, vid.height = crop_w, crop_h
            x, y, w, h = crop_xywh
            cropopt = f'crop={w}:{h}:{x}:{y}'
        else:
            crop_w, crop_h = vid.origin_width, vid.origin_height
            cropopt = ''

        vid.crop_width, vid.crop_height = crop_w, crop_h

        if resize is None or resize == (vid.crop_width, vid.crop_height):
            scaleopt = ''
            padopt = ''
        else:
            vid.width, vid.height = dst_width, dst_height = resize
            if not resize_keepratio:
                scaleopt = f'scale={dst_width}x{dst_height}'
                padopt = ''
            else:
                re_width, re_height = crop_w/(crop_h / dst_height) , dst_height
                if re_width > dst_width:
                    re_width, re_height = dst_width, crop_h/(crop_w / dst_width)
                re_width, re_height = int(re_width), int(re_height)
                scaleopt = f'scale={re_width}x{re_height}'
                if resize_keepratioalign is None: resize_keepratioalign = 'center'
                paddings = {'center': ((dst_width - re_width) // 2, (dst_height - re_height) // 2),
                            'topleft': (0, 0),
                            'topright': (dst_width - re_width, 0),
                            'bottomleft': (0, dst_height - re_height), 
                            'bottomright': (dst_width - re_width, dst_height - re_height)}
                assert resize_keepratioalign in paddings, 'resize_keepratioalign must be one of "center"(mmpose), "topleft"(mmdetection), "topright", "bottomleft", "bottomright"'
                xpading, ypading = paddings[resize_keepratioalign]
                padopt = f'pad={dst_width}:{dst_height}:{xpading}:{ypading}:black'
        
        if any([cropopt, scaleopt, padopt]):
            filterstr = ','.join(x for x in [cropopt, scaleopt, padopt] if x)
            filteropt = f'-vf {filterstr}'
        else:
            filteropt = ''

        vid.filteropt = filteropt
        return vid

    def __getitem__(self, indexs_org):
        if isinstance(indexs_org, int):
            indexs = [indexs_org]
        elif isinstance(indexs_org, (tuple, list)):
            indexs = indexs_org
        else:
            raise TypeError('indexs must be int or list or tuple')
        assert len(indexs), 'indexs must be nonempty'
        assert all([isinstance(x, int) for x in indexs]), 'The indexs should be intergers'
        assert all([x in range(self.count) for x in indexs]), 'The indexs should be in range'
        
        if max(indexs) + self.fps < self.count:
            time_to = int((max(indexs)) / self.fps)+1 #seconds
            time_to = time.strftime('%H:%M:%S', time.gmtime(time_to)) #str
            time_toopt = f'-to {time_to}'
        else:
            time_toopt = ''
        
        # if min(indexs) - self.fps > 1:
        #     time_from = int((min(indexs)) / self.fps)-1
        #     time_from = time.strftime('%H:%M:%S', time.gmtime(time_from))
        #     time_fromopt   = f'-ss {time_from}'
        # else:
        #     time_fromopt = ''

        time_fromopt = ''

        index_vobose = indexs + [indexs[-1]+1]
        eqlist = [r'eq(n\,%d)' % x for x in index_vobose]
        selectopt = "select='{}'".format( '+'.join(eqlist))
        if self.filteropt:
            filteropt = self.filteropt + ',' + selectopt
        else:
            filteropt = '-vf ' + selectopt
        args = (f'ffmpeg -loglevel warning '
                f' {time_fromopt} {time_toopt} '
                f' -vcodec {self.codec} -r {self.fps} -i "{self.filename}" '
                f' {filteropt} -pix_fmt {self.pix_fmt} -vsync drop -f rawvideo pipe:')
        self.process = run_async(args)
   
        imglist = [self._read()[1] for _ in indexs]
        # assert all(img is not None for img in imglist), 'Fail to _read frames'
        self._release()

        if isinstance(indexs_org, int):
            return imglist[0]
        else:
            return dict(zip(indexs, imglist))

    def _read(self):
        in_bytes = self.process.stdout.read(self.height * self.width * 3)
        time.sleep(0.2)
        if not in_bytes:
            self._release()
            return False, None
        img = None
        img = np.frombuffer(in_bytes, np.uint8).reshape([self.height, self.width, 3])
        return True, img

    def _release(self):
        if hasattr(self, 'process'):
            release_process(self.process)
