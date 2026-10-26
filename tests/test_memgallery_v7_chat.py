import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from scripts import run_memgallery_v7_chat as runner
from agent_memory.memgallery_v7_reader import pack_memgallery_evidence


class V7ChatReceiptProjectionTests(unittest.TestCase):
    def fixture(self, value):
        identity = '["synthetic-session","synthetic-round"]'
        query = dict(index=0, question='What color is the synthetic object?', point='FR')
        source = dict(text='The synthetic object is blue.', source_session='synthetic-session',
                      source_round='synthetic-round', source_date='2026-01-01', image_ids=[])
        pack = pack_memgallery_evidence(query, [identity], {identity: source}, lambda s: len(s)//4+1)
        request = dict(request_id='synthetic', metadata=dict(query_id='memgallery-v7:0', index=0,
                       point='FR', variant='dense', pack=pack))
        response = dict(model='gpt-4o-mini', choices=[dict(finish_reason='stop', message=dict(content=json.dumps(value)))])
        result = dict(response=response, attempt_id='synthetic-attempt', receipt_sha256='a'*64, run_ordinal=1)
        return request, result

    def test_auxiliary_fields_do_not_invalidate_valid_substantive_response(self):
        value = dict(answer='blue', cited_ids=['["synthetic-session","synthetic-round"]'],
                     confidence='supported', summary='A short answer.', evidence_quotes=['blue'])
        request, result = self.fixture(value)
        row = runner.saved_row(request, result, 'reader')
        self.assertEqual(row['answer']['schema_status'], 'valid')
        self.assertEqual(row['answer']['answer'], 'blue')
        self.assertEqual(row['receipt_sha256'], 'a'*64)

    def test_unknown_citation_is_invalid_without_substantive_repair(self):
        value = dict(answer='blue', cited_ids=['invented'], confidence='supported', summary='', evidence_quotes=[])
        request, result = self.fixture(value)
        before = json.dumps(result, sort_keys=True)
        row = runner.saved_row(request, result, 'reader')
        self.assertEqual(row['answer']['schema_status'], 'invalid')
        self.assertEqual(row['answer']['answer'], '')
        self.assertEqual(json.dumps(result, sort_keys=True), before)

    def test_invalid_invocation_limit_fails_before_filesystem_or_provider(self):
        with patch.object(runner, 'singleton', side_effect=AssertionError('must not create guard')):
            for limit in (0, -1, True, 1.2):
                with self.assertRaises(ValueError): runner.execute('reader', limit)


if __name__ == '__main__': unittest.main()
