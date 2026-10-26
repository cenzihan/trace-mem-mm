import pathlib,subprocess,time,os,json
p=pathlib.Path(__file__).resolve().parent
args=['/usr/bin/ssh','-F','/dev/null','-p','22422','-N','-T','-i',str(p/'tunnel_ed25519'),'-o','KexAlgorithms=ecdh-sha2-nistp256','-c','aes128-ctr','-o','ConnectTimeout=30','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+str(p/'server_known_hosts'),'-o','ExitOnForwardFailure=yes','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','-R','127.0.0.1:18978:127.0.0.1:18978','tracemem-reranker@45.192.104.173']
expiry=1793896200;attempt=0
while time.time()<expiry:
 attempt+=1
 with (p/'tunnel-stable.log').open('ab') as log:
  child=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=log,stderr=log)
  (p/'tunnel-current.json').write_text(json.dumps({'pid':child.pid,'supervisor_pid':os.getpid(),'connect_attempt':attempt}))
  try:child.wait(timeout=max(1,expiry-time.time()))
  except subprocess.TimeoutExpired:child.terminate();break
 time.sleep(min(60,5*attempt))
