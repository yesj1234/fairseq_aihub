import struct
import os 
import pathlib
from tqdm import tqdm 
import time
def make_wav_format(pcm_data:bytes, ch:int) -> bytes:
    """ 
    pcm_data를 통해서 wav 헤더를 만들고 wav 형식으로 저장한다.
    :param pcm_data: pcm bytes
    :param ch: 채널 수
    :return wav: wave bytes
    """
    waves = []
    waves.append(struct.pack('<4s', b'RIFF'))
    waves.append(struct.pack('I', 1))  
    waves.append(struct.pack('4s', b'WAVE'))
    waves.append(struct.pack('4s', b'fmt '))
    waves.append(struct.pack('I', 16))
    # audio_format, channel_cnt, sample_rate, bytes_rate(sr*blockalign:초당 바이츠수), block_align, bps
    if ch == 2:
        waves.append(struct.pack('HHIIHH', 1, 2, 16000, 64000, 4, 16))  
    else:
        waves.append(struct.pack('HHIIHH', 1, 1, 16000, 32000, 2, 16))
    waves.append(struct.pack('<4s', b'data'))
    waves.append(struct.pack('I', len(pcm_data)))
    waves.append(pcm_data)
    waves[1] = struct.pack('I', sum(len(w) for w in waves[2:]))
    return b''.join(waves)


def delete_pcm(source):
    for root, dir, files in tqdm(os.walk(source)):
        if files:
            for file in files:
                file_name, ext = os.path.splitext(file)
                if ext == ".pcm":
                    os.remove(os.path.join(root, file))

def main():
    source_path="C:\\Users\\must_\\Downloads\\kspon\\평가용_데이터"
    for root, dir, files in tqdm(os.walk(source_path), desc="Progress: "):
        if files:
            print(root)
            for file in files:
                file_name, ext = os.path.splitext(file)
                if ext == '.pcm':
                    pcm_bytes = pathlib.Path(os.path.join(root, file)).read_bytes()
                    wav_bytes = make_wav_format(pcm_bytes, 1)
                    with open(f'{os.path.join(root, file_name)}.wav', "wb") as wav:
                        wav.write(wav_bytes)
    time.sleep(0.5)
    delete_pcm(source=source_path)

    
if __name__ =="__main__":
    main()