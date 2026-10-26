import json
import unittest
from agent_memory.privacy_audit import sensitivity_counts


class PrivacyAuditTests(unittest.TestCase):
    def test_counts_never_return_matched_values(self):
        result=sensitivity_counts([dict(detail='person@synthetic.invalid payment password',location_name='UNIT_PLACE',timestamp='UNIT_TIME'),dict(detail='plain fixture')])
        self.assertEqual(result['records'],2);self.assertEqual(result['any_lexical_signal'],1)
        self.assertEqual(result['records_with_signals']['email_address_pattern'],1)
        self.assertEqual(result['transfer_cleared_records'],0)
        text=json.dumps(result);self.assertNotIn('person@',text);self.assertNotIn('UNIT_PLACE',text)

    def test_nested_signals_and_no_hit_not_clearance(self):
        result=sensitivity_counts([dict(nested=[dict(text='[phone_number] hospital child')]),dict(detail='plain')])
        self.assertEqual(result['records_with_signals']['health_terms'],1)
        self.assertEqual(result['records_with_signals']['redaction_marker'],1)
        self.assertTrue(result['no_hit_does_not_mean_safe']);self.assertEqual(result['transfer_cleared_records'],0)


if __name__=='__main__':unittest.main()
