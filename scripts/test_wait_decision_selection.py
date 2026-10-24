import unittest

from wait_decision_selection import compare


def row(rid, phase, truth, baseline, candidate, capture='capture-a', model='llama'):
    return dict(capture=capture, model=model, phase=phase, request_id=rid,
                truth=truth, predictions=dict(base=baseline, candidate=candidate))


class SelectionTests(unittest.TestCase):
    def test_tradeoff_is_rejected_without_a_cost_assumption(self):
        result = compare([row('late', 'pressure', True, False, True),
                          row('healthy', 'recovery', False, False, True)], 'base', 'candidate')
        self.assertFalse(result['eligible'])
        self.assertEqual(result['violations'][0]['reason'], 'fp increased')

    def test_improvement_cannot_hide_another_capture_regression(self):
        rows = [row('a', 'pressure', True, False, True),
                row('b', 'pressure', True, True, False, capture='capture-b')]
        self.assertFalse(compare(rows, 'base', 'candidate')['eligible'])

    def test_individual_pressure_phases_cannot_cancel_each_other(self):
        rows = [row('a', 'pressure', True, False, True),
                row('b', 'pressure-high', True, True, False)]
        result = compare(rows, 'base', 'candidate')
        self.assertFalse(result['eligible'])
        self.assertTrue(any(c['phase'] == 'overload_combined' for c in result['cells']))

    def test_equal_unknown_count_with_different_unknown_requests_fails(self):
        rows = [row('a', 'pressure', True, False, None),
                row('b', 'pressure', True, None, True)]
        result = compare(rows, 'base', 'candidate')
        self.assertFalse(result['eligible'])
        self.assertIn('prediction coverage differs', [r['reason'] for r in result['violations']])

    def test_unchanged_and_unknown_truth_are_not_improvements(self):
        rows = [row('a', 'recovery', False, False, False),
                row('rejected', 'recovery', None, True, False)]
        result = compare(rows, 'base', 'candidate')
        self.assertFalse(result['eligible'])
        self.assertEqual(result['reason'], 'no_strict_improvement')
        self.assertEqual(result['cells'][0]['candidate']['truth_unknown'], 1)

    def test_real_improvement_preserves_undefined_rate_denominator(self):
        result = compare([row('a', 'pressure', True, False, True)], 'base', 'candidate')
        self.assertTrue(result['eligible'])
        self.assertEqual(result['cells'][0]['candidate']['healthy_denominator'], 0)

    def test_empty_duplicate_and_numeric_flags_are_invalid(self):
        for rows in ([], [row('a', 'pressure', True, False, True)] * 2,
                     [row('a', 'pressure', True, False, 1)]):
            with self.assertRaises(ValueError):
                compare(rows, 'base', 'candidate')


if __name__ == '__main__':
    unittest.main()
