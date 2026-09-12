import sys, os, site
from pathlib import Path

# Simulate the same path logic
if sys.platform == 'win32':
    for sp in site.getsitepackages():
        nvidia_dir = Path(sp) / 'nvidia'
        if nvidia_dir.exists():
            for bin_dir in nvidia_dir.rglob('bin'):
                os.environ['PATH'] = str(bin_dir) + ';' + os.environ.get('PATH', '')
                try: os.add_dll_directory(str(bin_dir))
                except: pass

import paddle
import concurrent.futures

def _init_worker(use_gpu):
    global _reader
    import torch
    import easyocr
    gpu = use_gpu and torch.cuda.is_available()
    _reader = easyocr.Reader(['en'], gpu=gpu, verbose=False)

def _do_work(img_path):
    global _reader
    return len(_reader.readtext(img_path))

if __name__ == '__main__':
    print('Paddle GPU:', paddle.device.is_compiled_with_cuda())
    with concurrent.futures.ProcessPoolExecutor(max_workers=1, initializer=_init_worker, initargs=(True,)) as executor:
        f = executor.submit(_do_work, '../../training2/gstr2a_bw/jpg/GSTR2A_0002.jpg')
        print('EasyOCR text chunk count:', f.result())
