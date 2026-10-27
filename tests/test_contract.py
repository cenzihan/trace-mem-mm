import copy
import hashlib
import io
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from agent_memory.adapters import ModelConfig
from agent_memory.app import Application
from agent_memory.protocol import ContractError, IMAGE_LIMIT, MEDIA_LIMIT, image_bytes, validate_add
from agent_memory.store import MemoryStore

ROOT = Path(__file__).resolve().parent.parent


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dbpath = Path(self.temp.name) / "memory.sqlite"
        self.store = MemoryStore(self.dbpath)
        self.add = json.loads((ROOT / "fixtures/add.json").read_text(encoding="utf-8"))
        self.query = json.loads((ROOT / "fixtures/search.json").read_text(encoding="utf-8"))

    def tearDown(self):
        self.temp.cleanup()

    def call(self, path, payload=None, method="POST", app=None, **headers):
        raw = json.dumps(payload).encode() if payload is not None else b""
        env = dict(PATH_INFO=path, REQUEST_METHOD=method, CONTENT_TYPE="application/json",
                   CONTENT_LENGTH=str(len(raw)), **{"wsgi.input": io.BytesIO(raw)}, **headers)
        seen = []
        response = b"".join((app or Application(self.store))(env, lambda status, h: seen.append(status)))
        return int(seen[0].split()[0]), json.loads(response)

    def test_add_exact_echo_and_immediate_search(self):
        status, result = self.call("/add", self.add)
        self.assertEqual(status, 200)
        self.assertEqual(result, {k: self.add[k] for k in ("request_id", "user_id", "session_id")} | {"success": True})
        self.assertEqual(len(self.call("/search", self.query)[1]["data"]), 2)

    def test_original_order_image_and_source_time_round_trip(self):
        self.store.add(self.add)
        top = self.store.search(self.query)["data"][0]
        self.assertEqual(top["content"], self.add["messages"][0]["content"])
        self.assertEqual(top["created_at"], "2024-01-01T00:00:00Z")

    def test_persistence_across_reopen(self):
        self.store.add(self.add)
        before = self.store.search(self.query)
        self.assertEqual(before, MemoryStore(self.dbpath).search(self.query))

    def test_idempotent_retry(self):
        first = self.store.add(self.add)
        self.assertEqual(first, self.store.add(copy.deepcopy(self.add)))
        self.assertEqual(len(self.store.search(self.query)["data"]), 2)

    def test_concurrent_retry_only_one_write(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: self.store.add(copy.deepcopy(self.add)), range(4)))
        self.assertTrue(all(result == results[0] for result in results))
        self.assertEqual(len(self.store.search(self.query)["data"]), 2)

    def test_conflicting_retry_rolls_back(self):
        self.store.add(self.add)
        self.add["messages"][0]["content"][0]["text"] = "changed"
        self.assertEqual(self.call("/add", self.add)[0], 409)
        self.assertNotIn("changed", json.dumps(self.store.search(self.query)))

    def test_unknown_user_empty(self):
        self.store.add(self.add)
        self.query["user_id"] = "synthetic:user:unknown"
        self.assertEqual(self.store.search(self.query), {"data": []})

    def test_same_ids_different_users_isolated(self):
        self.store.add(self.add)
        other = copy.deepcopy(self.add)
        other["user_id"] = "synthetic:user:bob"
        self.store.add(other)
        a = {r["id"] for r in self.store.search(self.query)["data"]}
        self.query["user_id"] = other["user_id"]
        b = {r["id"] for r in self.store.search(self.query)["data"]}
        self.assertFalse(a & b)

    def test_session_provenance_separate_same_user_searches_both(self):
        self.store.add(self.add)
        self.add["session_id"], self.add["request_id"] = "session:two", "request:two"
        self.store.add(self.add)
        self.assertEqual(len(self.store.search(self.query)["data"]), 4)
        with self.store.connect() as db:
            self.assertEqual(db.execute("SELECT count(DISTINCT session_id) FROM memories").fetchone()[0], 2)
            # Identical image bytes do not merge event occurrences.
            self.assertEqual(db.execute("SELECT count(*) FROM evidence").fetchone()[0], 4)

    def test_session_not_accepted_as_search_filter(self):
        self.query["session_id"] = self.add["session_id"]
        self.assertEqual(self.call("/search", self.query)[0], 422)

    def test_stable_ranking_ties_and_top_k(self):
        self.store.add(self.add)
        self.query["query"] = "unrelated"
        data = self.store.search(self.query)["data"]
        self.assertEqual([d["id"] for d in data], sorted(d["id"] for d in data))
        self.query["top_k"] = 1
        self.assertEqual(self.store.search(self.query)["data"], data[:1])

    def test_exact_image_only_retrieval(self):
        self.store.add(self.add)
        self.query["query"] = [self.add["messages"][1]["content"][1]]
        data = self.store.search(self.query)["data"]
        self.assertEqual(data[0]["content"], self.add["messages"][1]["content"])
        self.assertGreater(data[0]["score"], data[1]["score"])

    def test_no_query_or_option_persistence(self):
        self.store.add(self.add)
        before = self.dbpath.read_bytes()
        self.query["query"] = "PRIVATE_QUERY_SENTINEL"
        self.query["options"] = ["PRIVATE_OPTION_SENTINEL"]
        self.store.search(self.query)
        self.assertEqual(before, self.dbpath.read_bytes())

    def test_evidence_hash_and_occurrence_links(self):
        self.store.add(self.add)
        with self.store.connect() as db:
            for sha, raw, ordinal in db.execute("SELECT sha256, original, part_ordinal FROM evidence"):
                self.assertEqual(sha, hashlib.sha256(raw).hexdigest())
                self.assertEqual(ordinal, 1)

    def test_plain_string_normalized_to_multimodal_text(self):
        self.add["messages"] = [{"role": "user", "content": "synthetic text"}]
        self.store.add(self.add)
        self.assertEqual(self.store.search(self.query)["data"][0]["content"], [{"type": "text", "text": "synthetic text"}])

    def test_malformed_add_is_atomic(self):
        self.add["messages"].append({"role": "user", "content": []})
        self.assertEqual(self.call("/add", self.add)[0], 422)
        self.assertEqual(self.store.search(self.query), {"data": []})

    def test_invalid_add_fields(self):
        for key, value in (("request_id", ""), ("user_id", None), ("session_id", 42), ("messages", [])):
            with self.subTest(key=key):
                p = copy.deepcopy(self.add)
                p[key] = value
                self.assertEqual(self.call("/add", p)[0], 422)

    def test_invalid_message_fields(self):
        for msg in ({"role": "system", "content": "text"}, {"role": "user", "content": " "},
                    {"role": "user", "content": "text", "timestamp": True},
                    {"role": "user", "content": "text", "timestamp": 10**30}):
            with self.subTest(msg=msg):
                p = copy.deepcopy(self.add)
                p["messages"] = [msg]
                self.assertEqual(self.call("/add", p)[0], 422)

    def test_invalid_parts_remote_base64_mime(self):
        for part in ({"type": "video"}, {"type": "text", "text": ""},
                     {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
                     {"type": "image_url", "image_url": {"url": "data:image/png;base64,???"}},
                     {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}}):
            with self.subTest(part=part):
                p = copy.deepcopy(self.add)
                p["messages"][0]["content"] = [part]
                self.assertEqual(self.call("/add", p)[0], 422)

    def test_image_and_add_media_limits(self):
        uri = self.add["messages"][0]["content"][1]["image_url"]["url"]
        size = len(image_bytes(uri)[0])
        # Exercise boundary accounting without repeatedly allocating 30 MiB test blobs.
        with patch("agent_memory.protocol.IMAGE_LIMIT", size):
            image_bytes(uri)
        with patch("agent_memory.protocol.IMAGE_LIMIT", size - 1):
            self.assertRaises(ContractError, image_bytes, uri)
        total = sum(len(image_bytes(m["content"][1]["image_url"]["url"])[0]) for m in self.add["messages"])
        with patch("agent_memory.protocol.MEDIA_LIMIT", total):
            validate_add(self.add)
        with patch("agent_memory.protocol.MEDIA_LIMIT", total - 1):
            self.assertRaises(ContractError, validate_add, self.add)
        self.assertEqual((IMAGE_LIMIT, MEDIA_LIMIT), (10485760, 31457280))

    def test_search_media_capacity_clear_error_not_truncation(self):
        self.store.add(self.add)
        with patch("agent_memory.store.MEDIA_LIMIT", 1):
            self.assertEqual(self.call("/search", self.query)[0], 422)

    def test_invalid_search_options_and_top_k(self):
        for field, value in (("top_k", 0), ("top_k", True), ("top_k", 1.5), ("options", "A"),
                             ("options", [1]), ("query", [])):
            with self.subTest(field=field, value=value):
                p = copy.deepcopy(self.query)
                p[field] = value
                self.assertEqual(self.call("/search", p)[0], 422)

    def test_open_question_options_optional(self):
        del self.query["options"]
        self.assertEqual(self.call("/search", self.query), (200, {"data": []}))

    def test_auth_modes_and_unauthenticated_health(self):
        for mode in ("Token", "Bearer", "X-Api-Key"):
            with self.subTest(mode=mode):
                app = Application(self.store, mode, "synthetic-ephemeral-key")
                self.assertEqual(self.call("/health", method="GET", app=app)[0], 200)
                self.assertEqual(self.call("/search", self.query, app=app)[0], 401)
                headers = ({"HTTP_X_API_KEY": "synthetic-ephemeral-key"} if mode == "X-Api-Key" else
                           {"HTTP_AUTHORIZATION": mode + " synthetic-ephemeral-key"})
                self.assertEqual(self.call("/search", self.query, app=app, **headers)[0], 200)
                self.assertEqual(self.call("/search", self.query, app=app, HTTP_AUTHORIZATION="wrong")[0], 401)

    def test_routes_and_local_body_capacity(self):
        self.assertEqual(self.call("/add", method="GET")[0], 405)
        self.assertEqual(self.call("/missing")[0], 404)
        self.assertEqual(self.call("/add", self.add, app=Application(self.store, max_body=1))[0], 413)

    def test_invalid_json_no_content_logging(self):
        env = {"PATH_INFO": "/add", "REQUEST_METHOD": "POST", "CONTENT_TYPE": "application/json",
               "CONTENT_LENGTH": "4", "wsgi.input": io.BytesIO(b"oops")}
        seen = []
        body = b"".join(Application(self.store)(env, lambda status, h: seen.append(status)))
        self.assertTrue(seen[0].startswith("400"))
        self.assertNotIn(b"oops", body)

    def test_model_policy_fails_closed(self):
        for config in (ModelConfig(embedding_model="CLIP"), ModelConfig(llm_model="Qwen3-VL-2B"),
                       ModelConfig(reranker="Qwen/Qwen3-VL-Reranker-2B")):
            with self.subTest(config=config):
                self.assertRaises(ValueError, MemoryStore, self.dbpath, config)
        self.assertRaises(RuntimeError, MemoryStore, self.dbpath, ModelConfig(mode="configured"))

    def test_reranker_extension_bad_scores_fail(self):
        class BadReranker:
            def score(self, *args):
                return [float("nan")]
        store = MemoryStore(self.dbpath, reranker=BadReranker())
        store.add(self.add)
        self.assertEqual(self.call("/search", self.query, app=Application(store))[0], 500)

    def test_purge_removes_only_one_user_and_evidence(self):
        self.store.add(self.add)
        other = copy.deepcopy(self.add)
        other["user_id"] = "synthetic:user:bob"
        self.store.add(other)
        self.store.purge_user(self.add["user_id"])
        self.assertEqual(self.store.search(self.query), {"data": []})
        self.query["user_id"] = other["user_id"]
        self.assertEqual(len(self.store.search(self.query)["data"]), 2)
        with self.store.connect() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM evidence").fetchone()[0], 2)


if __name__ == "__main__":
    unittest.main()
