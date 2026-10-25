"""Local CLI. No model calls, official evaluations or network listeners by default."""
import argparse
import ipaddress
import json
from pathlib import Path

from .app import Application
from .system import AgentMemorySystem,RetrievalConfig


def main(argv=None):
    parser=argparse.ArgumentParser(description="Academic multimodal memory: local deterministic mode")
    parser.add_argument("--db",default="runtime/system.sqlite")
    parser.add_argument("--no-temporal",action="store_true")
    parser.add_argument("--no-links",action="store_true")
    sub=parser.add_subparsers(dest="action",required=True)
    for name in ("add","search"):
        p=sub.add_parser(name); p.add_argument("input",type=Path)
    p=sub.add_parser("inspect"); p.add_argument("user_id")
    sub.add_parser("capabilities")
    sub.add_parser("integrity")
    p=sub.add_parser("repair-index");p.add_argument("--user-id")
    p=sub.add_parser("backup");p.add_argument("destination",type=Path)
    p=sub.add_parser("serve"); p.add_argument("--host",default="127.0.0.1"); p.add_argument("--port",default=8765,type=int)
    p.add_argument("--development-loopback",action="store_true",required=True)
    args=parser.parse_args(argv)
    if args.action in ('integrity','repair-index','backup'):
        from .maintenance import integrity,repair_missing_indexes,backup_database
        if args.action=='integrity':output=integrity(args.db)
        elif args.action=='repair-index':output=repair_missing_indexes(args.db,args.user_id)
        else:output=backup_database(args.db,args.destination)
        print(json.dumps(output,indent=2));return
    store=AgentMemorySystem(args.db,RetrievalConfig(temporal=not args.no_temporal,links=not args.no_links))
    if args.action in ("add","search"):
        payload=json.loads(args.input.read_text(encoding="utf-8-sig"))
        output=getattr(store,args.action)(payload)
    elif args.action=="inspect": output=store.inspect_user(args.user_id)
    elif args.action=="capabilities": output=store.capabilities()
    else:
        if not ipaddress.ip_address(args.host).is_loopback:
            raise ValueError("public binding is not authorized; loopback only")
        from wsgiref.simple_server import make_server,WSGIRequestHandler
        class NoBodyLogging(WSGIRequestHandler):
            def log_message(self,*args): pass
        print("Development loopback server only; no official evaluation or model service configured.",flush=True)
        with make_server(args.host,args.port,Application(store),handler_class=NoBodyLogging) as server:
            server.serve_forever()
        return
    print(json.dumps(output,ensure_ascii=False,indent=2))
