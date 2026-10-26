"""Container-local read-only readiness probe; no model invocation."""
import json
import urllib.request

with urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3) as response:
    value = json.load(response)
    if response.status != 200 or value.get('status') != 'ok':
        raise SystemExit(1)
