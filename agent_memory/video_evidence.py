"""Bounded local video frames with original evidence identity; no model/network calls.

Requires separately verified local ffmpeg/ffprobe binaries. Frame sampling is a
declared approximation, never proof of complete temporal or audio coverage.
"""
import hashlib
import json
import math
import queue
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path,PurePosixPath


@dataclass(frozen=True)
class FramePolicy:
    frames:int=6
    long_side:int=1024
    max_duration_seconds:int=3600
    max_source_bytes:int=512*1024*1024
    process_timeout_seconds:int=15
    def validate(self):
        for value,low,high in ((self.frames,1,8),(self.long_side,64,1024),(self.max_duration_seconds,1,3600),
            (self.max_source_bytes,1,512*1024*1024),(self.process_timeout_seconds,1,30)):
            if type(value) is not int or not low<=value<=high:raise ValueError('bounded frame policy required')


def verified_video(root,relative,expected_sha256,policy=FramePolicy()):
    policy.validate()
    if not isinstance(relative,str) or '\\' in relative or ':' in relative:raise ValueError('local native video reference required')
    name=PurePosixPath(relative)
    if name.is_absolute() or '..' in name.parts or name.suffix.lower()!='.mp4':raise ValueError('invalid native video path')
    if not isinstance(expected_sha256,str) or len(expected_sha256)!=64 or any(c not in '0123456789abcdef' for c in expected_sha256):
        raise ValueError('verified source SHA256 required')
    base=Path(root).resolve();path=(base/relative).resolve()
    if not path.is_relative_to(base) or not path.is_file():raise ValueError('video escapes local reviewed root')
    if path.stat().st_size>policy.max_source_bytes:raise ValueError('source video exceeds bounded admission')
    digest=hashlib.sha256()
    with path.open('rb') as source:
        head=source.read(32);digest.update(head)
        if len(head)<12 or head[4:8]!=b'ftyp':raise ValueError('MP4 signature required')
        for block in iter(lambda:source.read(1024*1024),b''):digest.update(block)
    if digest.hexdigest()!=expected_sha256:raise ValueError('native video source hash mismatch')
    return path


def bounded_process(command,timeout,max_bytes):
    """No shell, bounded stdout, no stderr/source paths in errors, kill on expiry."""
    if not isinstance(command,list) or not command or not Path(command[0]).is_absolute() or not Path(command[0]).is_file():
        raise ValueError('verified absolute local media tool required')
    process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,shell=False)
    result=queue.Queue(maxsize=1)
    def read():
        try:result.put(process.stdout.read(max_bytes+1))
        except Exception:result.put(None)
    thread=threading.Thread(target=read,daemon=True);thread.start()
    try:
        raw=result.get(timeout=timeout)
        if not isinstance(raw,bytes) or len(raw)>max_bytes:raise ValueError('media subprocess output limit')
        code=process.wait(timeout=1)
        if code:raise ValueError('media subprocess failed')
        return raw
    except (queue.Empty,subprocess.TimeoutExpired):raise ValueError('media subprocess deadline exceeded') from None
    finally:
        if process.poll() is None:process.kill()
        try:process.wait(timeout=2)
        except subprocess.TimeoutExpired:pass
        process.stdout.close()


def extract_video_frames(root,relative,expected_sha256,evidence_id,*,ffprobe,ffmpeg,runner=bounded_process,policy=FramePolicy()):
    path=verified_video(root,relative,expected_sha256,policy)
    if not isinstance(evidence_id,str) or not evidence_id:raise ValueError('native video evidence ID required')
    probe=[str(ffprobe),'-v','error','-protocol_whitelist','file','-print_format','json',
           '-show_entries','format=duration:stream=codec_type',str(path)]
    metadata=json.loads(runner(probe,policy.process_timeout_seconds,65536))
    streams=metadata.get('streams');duration=float(metadata.get('format',{}).get('duration','nan'))
    if not isinstance(streams,list) or not any(s.get('codec_type')=='video' for s in streams):raise ValueError('native video stream required')
    if any(s.get('codec_type')=='audio' for s in streams):raise ValueError('audio present: frame-only path cannot claim complete video support')
    if not math.isfinite(duration) or not 0<duration<=policy.max_duration_seconds:raise ValueError('video duration outside bounded admission')
    frames=[]
    for ordinal in range(policy.frames):
        seconds=duration*(ordinal+.5)/policy.frames
        command=[str(ffmpeg),'-nostdin','-hide_banner','-loglevel','error','-threads','1',
                 '-protocol_whitelist','file,pipe','-ss',f'{seconds:.6f}','-i',str(path),'-map','0:v:0',
                 '-frames:v','1','-an','-sn','-dn','-vf',f'scale={policy.long_side}:{policy.long_side}:force_original_aspect_ratio=decrease',
                 '-threads','1','-f','image2pipe','-vcodec','png','pipe:1']
        raw=runner(command,policy.process_timeout_seconds,4*1024*1024)
        if len(raw)<24 or not raw.startswith(b'\x89PNG\r\n\x1a\n') or raw[12:16]!=b'IHDR':raise ValueError('decoded PNG frame required')
        width=int.from_bytes(raw[16:20],'big');height=int.from_bytes(raw[20:24],'big')
        if not 1<=max(width,height)<=policy.long_side or min(width,height)<1:raise ValueError('bounded frame dimensions required')
        frames.append(dict(native_evidence_id=evidence_id,frame_occurrence_id=evidence_id+':frame:'+str(ordinal),
            frame_offset_seconds=seconds,source_video_sha256=expected_sha256,frame_sha256=hashlib.sha256(raw).hexdigest(),
            bytes=raw,width=width,height=height))
    return dict(native_evidence_id=evidence_id,source_video_sha256=expected_sha256,frames=frames,
        native_evidence_count=1,duration_seconds=duration,audio_present=False,audio_dropped=False,
        sampling='uniform midpoint frames, no QA/gold selection',complete_temporal_coverage=False,
        source_timestamp_or_GPS_inferred=False,model_calls=0)
