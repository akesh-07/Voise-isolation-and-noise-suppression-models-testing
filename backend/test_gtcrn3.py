import urllib.request
import json
import base64
import time

with open(r'C:\ShellKode\Projects\Practice\Open-src-models-test\backend\ClearerVoice-Studio\clearvoice\samples\speech1.wav', 'rb') as f:
    wav_bytes = f.read()

boundary = 'WebKitFormBoundary'
data = (
    f'--{boundary}\r\n'
    f'Content-Disposition: form-data; name="processing_mode"\r\n\r\n'
    f'gtcrn\r\n'
    f'--{boundary}\r\n'
    f'Content-Disposition: form-data; name="file"; filename="speech1.wav"\r\n'
    f'Content-Type: audio/wav\r\n\r\n'
).encode('utf-8') + wav_bytes + f'\r\n--{boundary}--\r\n'.encode('utf-8')

req = urllib.request.Request(
    'http://127.0.0.1:8000/process',
    data=data,
    headers={'Content-Type': 'multipart/form-data; boundary=' + boundary}
)

start = time.time()
try:
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode())
        with open('debug_out.txt', 'w') as f:
            f.write(json.dumps({'success': res.get('success')}))
except urllib.error.HTTPError as e:
    with open('debug_out.txt', 'w') as f:
        f.write(e.read().decode())
