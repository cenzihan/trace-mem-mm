"""WSGI transport. Importing/constructing this application never binds a port."""
import hmac
import json
from http import HTTPStatus

from .protocol import ContractError, canonical


class Application:
    def __init__(self, store, auth_mode="none", key=None, max_body=64 * 1024 * 1024):
        if auth_mode not in ("none", "Token", "Bearer", "X-Api-Key"):
            raise ValueError("unsupported authentication mode")
        if auth_mode != "none" and (not isinstance(key, str) or not key):
            raise ValueError("authentication requires an externally supplied non-empty key")
        self.store, self.auth_mode, self.key, self.max_body = store, auth_mode, key, max_body

    def authorized(self, environ):
        if self.auth_mode == "none":
            return True
        value = (environ.get("HTTP_X_API_KEY", "") if self.auth_mode == "X-Api-Key"
                 else environ.get("HTTP_AUTHORIZATION", ""))
        expected = self.key if self.auth_mode == "X-Api-Key" else self.auth_mode + " " + self.key
        return hmac.compare_digest(value.encode(), expected.encode())

    def record_exception(self, stage, exc, status, payload=None, response=None):
        """Optional private, content-free diagnostics supplied by the service."""
        pass

    def __call__(self, environ, start_response):
        status, path, payload = 200, None, None
        try:
            path, method = environ.get("PATH_INFO"), environ.get("REQUEST_METHOD")
            if path == "/health" and method == "GET":
                with self.store.connect() as db:
                    db.execute("SELECT 1 FROM memories LIMIT 1").fetchone()
                result = ({"status":"ok",**self.store.capabilities()} if hasattr(self.store,"capabilities") else
                          {"status": "ok", "mode": "mock", "baseline": "lexical+exact-image", "online_verified": False})
            elif path not in ("/add", "/search"):
                raise ContractError("route not found", 404)
            elif method != "POST":
                raise ContractError("POST required", 405)
            elif not self.authorized(environ):
                raise ContractError("invalid credentials", 401)
            else:
                if environ.get("CONTENT_TYPE", "").split(";")[0].strip().lower() != "application/json":
                    raise ContractError("application/json required", 415)
                try:
                    length = int(environ.get("CONTENT_LENGTH", ""))
                except (ValueError, TypeError) as exc:
                    raise ContractError("valid Content-Length required", 400) from exc
                if length < 0 or length > self.max_body:
                    raise ContractError("local HTTP capacity exceeded (64 MiB default)", 413)
                raw = environ["wsgi.input"].read(length)
                if len(raw) != length:
                    raise ContractError("incomplete body", 400)
                try:
                    payload = json.loads(raw.decode("utf-8"), parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
                except (UnicodeError, ValueError) as exc:
                    raise ContractError("invalid JSON", 400) from exc
                # Do not retain a second full body throughout model waits/commit.
                del raw
                result = self.store.add(payload) if path == "/add" else self.store.search(payload)
        except ContractError as exc:
            status, result = exc.status, {"error": str(exc)}
            self.record_exception(path if path in ('/add', '/search', '/health') else 'other', exc, status, payload, result)
        except Exception as exc:
            # No exception details or private request content in public responses.
            status, result = 500, {"error": "internal service error"}
            self.record_exception(path if path in ('/add', '/search', '/health') else 'other', exc, status, payload, result)
        body = canonical(result).encode("utf-8")
        start_response(f"{status} {HTTPStatus(status).phrase}",
                       [("Content-Type", "application/json; charset=utf-8"), ("Content-Length", str(len(body))),
                        ("Cache-Control", "no-store")])
        return [body]
