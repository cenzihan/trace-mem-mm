"""Single-process SQLite service; this config does not enable V6 inference."""
bind = '0.0.0.0:8000'
workers = 1
worker_class = 'gthread'
threads = 4
preload_app = False
worker_tmp_dir = '/tmp'
control_socket_disable = True
# Worker liveness / shutdown settings, NOT per-request deadlines or an SLA.
timeout = 1900
graceful_timeout = 1900
keepalive = 5
max_requests = 1000
max_requests_jitter = 100
accesslog = None
errorlog = '-'
loglevel = 'warning'
capture_output = False
