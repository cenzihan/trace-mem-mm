import os,sys,json,time,math,hashlib,subprocess,importlib.util,signal,hmac
from pathlib import Path
from http.server import BaseHTTPRequestHandler,HTTPServer
work=Path(__file__).resolve().parent
module_file=Path('/software/cenzihan/agent-memory-multimodal/atm-v6-rerank-20261025/run_atm_v6_remote.py')
assert hashlib.sha256(module_file.read_bytes()).hexdigest()=='6593881aeeef157793c50bcb1da6622f0eb918a6ff81e20dbbb102754c4eb9be'
spec=importlib.util.spec_from_file_location('frozen_reranker',module_file);r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
gpu=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
lock=r.exclusive_lock(r.REMOTE_ROOT/('v6-'+gpu+'.lock'));lock.__enter__()
r.check_gpu_idle(gpu);r.verify_model()
os.environ.update(CUDA_VISIBLE_DEVICES=gpu,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',TOKENIZERS_PARALLELISM='false')
sys.path[:0]=list(r.DEPENDENCIES)
import torch,transformers
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
assert torch.__version__=='2.2.2+cu121' and transformers.__version__=='4.57.1'
transformers.utils.logging.set_verbosity_error();transformers.utils.logging.disable_progress_bar()
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
assert torch.cuda.is_bf16_supported()
torch.cuda.set_per_process_memory_fraction(.85,0);torch.cuda.reset_peak_memory_stats(0)
model=Qwen3VLForConditionalGeneration.from_pretrained(str(r.MODEL_PATH),local_files_only=True,trust_remote_code=False,use_safetensors=True,torch_dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa').eval()
processor=AutoProcessor.from_pretrained(str(r.MODEL_PATH),local_files_only=True,trust_remote_code=False,use_fast=False,padding_side='left')
tokenizer=processor.tokenizer;tokenizer.padding_side='left';vocab=tokenizer.get_vocab()
assert vocab['yes']==9693 and vocab['no']==2152
head=model.get_output_embeddings().weight;direction=head[vocab['yes']].detach().float()-head[vocab['no']].detach().float()

key=(work/'reranker_token').read_text().strip()
started=time.monotonic();count=0;pairs_done=0
expiry=1793896200
if (work/'stats.json').exists():
 previous=json.loads((work/'stats.json').read_text());count=previous['requests'];pairs_done=previous['pairs']
 assert 0<=count<=2000 and 0<=pairs_done<=200000
identity={k:r.SPEC[k] for k in ('model','revision','weights_sha256')}
def stats():
 return dict(identity,requests=count,pairs=pairs_done,elapsed_seconds=round(time.monotonic()-started,2),generation=False,source_text_only=True)
def save_stats(state):
 temp=work/'stats.tmp'
 with temp.open('w') as stream:
  json.dump(dict(stats(),state=state),stream);stream.flush();os.fsync(stream.fileno())
 os.replace(temp,work/'stats.json')
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def setup(self):
  super().setup();self.connection.settimeout(190)
 def reply(self,status,value):
  raw=json.dumps(value,allow_nan=False).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
 def do_GET(self):
  if self.path!='/health':return self.reply(404,{'error':'route'})
  return self.reply(200,stats())
 def do_POST(self):
  global count,pairs_done
  if self.path!='/score':return self.reply(404,{'error':'route'})
  if not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+key):return self.reply(401,{'error':'auth'})
  try:
   length=int(self.headers.get('Content-Length','0'))
   if not 0<length<=2*1024*1024:raise ValueError('length')
   data=json.loads(self.rfile.read(length));q=data['query'];docs=data['documents']
   if set(data)!={'query','documents'} or not isinstance(q,str) or not q.strip() or len(q.encode())>8192:raise ValueError('query')
   if not isinstance(docs,list) or not 1<=len(docs)<=100 or any(not isinstance(d,str) or not d.strip() or len(d.encode())>16384 for d in docs):raise ValueError('documents')
   if count>=2000 or pairs_done+len(docs)>200000:raise ValueError('quota')
   count+=1
   save_stats('scoring')
   encoded=[r.fit_document(q,d,lambda v:processor.apply_chat_template(v,tokenize=False,add_generation_prompt=True),lambda v:tokenizer.encode(v,add_special_tokens=False)) for d in docs]
   values=[]
   # Greedy ordered batches: <=8 pairs and <=8192 padded tokens.
   pending=[];longest=0
   def infer(batch):
    inputs=tokenizer.pad({'input_ids':[x['input_ids'] for x in batch]},padding=True,return_tensors='pt').to(model.device)
    with torch.inference_mode():
     hidden=model.model(**inputs,use_cache=False,return_dict=True).last_hidden_state[:,-1]
     return torch.mv(hidden.float(),direction).cpu().tolist()
   for item in encoded:
    n=len(item['input_ids'])
    if pending and (len(pending)>=8 or max(longest,n)*(len(pending)+1)>8192):values.extend(infer(pending));pending=[];longest=0
    pending.append(item);longest=max(longest,n)
   if pending:values.extend(infer(pending))
   assert len(values)==len(docs) and all(math.isfinite(v) for v in values)
   pairs_done+=len(docs);save_stats('ready')
   self.reply(200,dict(identity,scores=values))
  except Exception as exc:
   (work/'failure.json').write_text(json.dumps({'error_type':type(exc).__name__,'requests':count,'pairs':pairs_done}))
   self.reply(503,{'error':'scoring_unavailable'})
server=HTTPServer(('127.0.0.1',18978),Handler);server.timeout=5
save_stats('ready')
(work/'pid').write_text(str(os.getpid()))
try:
 while time.time()<expiry:server.handle_request()
finally:server.server_close();lock.__exit__(None,None,None)
