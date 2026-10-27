"""Unit tests for Assignment 1 solution.py (Role 5: Integration & Test Lead).
Run with: python -m unittest -v test_solution.py
"""
import unittest
import numpy as np
import solution as sol


class TestRecordValidation(unittest.TestCase):
    def test_valid_records_pass(self):
        self.assertIsNone(sol.validate_records([{'state':'A','date':'2020-01-01'}], ('state','date')))

    def test_non_list_raises_type_error(self):
        with self.assertRaises(TypeError):
            sol.validate_records({'state':'A'}, ('state',))

    def test_empty_list_raises_value_error(self):
        with self.assertRaises(ValueError):
            sol.validate_records([], ('state',))

    def test_non_dictionary_record_raises_type_error(self):
        with self.assertRaises(TypeError):
            sol.validate_records([42], ('state',))

    def test_missing_key_raises_key_error(self):
        with self.assertRaises(KeyError):
            sol.validate_records([{'state':'A'}], ('state','date'))

    def test_duplicate_state_year_raises_value_error(self):
        rows = [{'state':'A','date':'2020-01-01'}, {'state':'A','date':'2020-12-31'}]
        with self.assertRaises(ValueError):
            sol.validate_records(rows, ('state','date'))


class TestIndexAndJoin(unittest.TestCase):
    def test_index_uses_state_and_year(self):
        rows = [{'state':'A','date':'2020-01-01','x':1}]
        self.assertEqual(sol.index_by_key(rows)[('A', 2020)]['x'], 1)

    def test_index_duplicate_raises(self):
        rows = [{'state':'A','date':'2020-01-01'}, {'state':'A','date':'2020-02-01'}]
        with self.assertRaises(ValueError):
            sol.index_by_key(rows)

    def test_join_returns_common_and_unmatched_keys(self):
        inc = [{'state':'A','date':'2020-01-01','income_mean':10}, {'state':'B','date':'2020-01-01','income_mean':20}]
        pov = [{'state':'A','date':'2020-06-01','poverty_absolute':2}, {'state':'C','date':'2020-01-01','poverty_absolute':3}]
        joined, inc_only, pov_only = sol.join_records(inc, pov)
        self.assertEqual(len(joined), 1)
        self.assertEqual(joined[0]['state'], 'A')
        self.assertEqual(inc_only, {('B', 2020)})
        self.assertEqual(pov_only, {('C', 2020)})

    def test_to_float_none_is_nan(self):
        self.assertTrue(np.isnan(sol.to_float(None)))

    def test_to_float_numeric_string(self):
        self.assertEqual(sol.to_float('12.5'), 12.5)

    def test_to_float_invalid_string_is_nan(self):
        self.assertTrue(np.isnan(sol.to_float('missing')))


class TestMatricesAndAnalysis(unittest.TestCase):
    def setUp(self):
        self.income = [
            {'state':'A','date':'2016-01-01','income_mean':100,'income_median':80},
            {'state':'A','date':'2019-01-01','income_mean':120,'income_median':90},
            {'state':'A','date':'2020-01-01','income_mean':130,'income_median':95},
            {'state':'A','date':'2022-01-01','income_mean':140,'income_median':100},
            {'state':'A','date':'2024-01-01','income_mean':150,'income_median':110},
            {'state':'B','date':'2016-01-01','income_mean':90,'income_median':70},
            {'state':'B','date':'2019-01-01','income_mean':100,'income_median':75},
            {'state':'B','date':'2020-01-01','income_mean':105,'income_median':78},
            {'state':'B','date':'2022-01-01','income_mean':110,'income_median':80},
            {'state':'B','date':'2024-01-01','income_mean':115,'income_median':85},
        ]
        self.poverty = [
            {'state':'A','date':'2019-01-01','poverty_absolute':5},
            {'state':'A','date':'2020-01-01','poverty_absolute':6},
            {'state':'A','date':'2022-01-01','poverty_absolute':4},
            {'state':'A','date':'2024-01-01','poverty_absolute':3},
            {'state':'B','date':'2019-01-01','poverty_absolute':7},
            {'state':'B','date':'2020-01-01','poverty_absolute':7},
            {'state':'B','date':'2022-01-01','poverty_absolute':8},
            {'state':'B','date':'2024-01-01','poverty_absolute':6},
        ]

    def test_build_matrices_shapes_and_state_order(self):
        states, median, mean, poverty = sol.build_matrices(self.income, self.poverty)
        self.assertEqual(states, ['A','B'])
        self.assertEqual(median.shape, (2,5))
        self.assertEqual(mean.shape, (2,5))
        self.assertEqual(poverty.shape, (2,4))

    def test_missing_year_is_nan(self):
        income = [r for r in self.income if not (r['state']=='A' and r['date'].startswith('2019'))]
        states, median, _, _ = sol.build_matrices(income, self.poverty)
        self.assertTrue(np.isnan(median[states.index('A'), 1]))

    def test_focus_state_vector_changes(self):
        states, median, _, _ = sol.build_matrices(self.income, self.poverty)
        abs_change, pct_change, annual, falling = sol.analyze_focus_state_vectors('A', states, sol.SURVEY_YEARS, median)
        np.testing.assert_allclose(abs_change, [10,5,5,10])
        self.assertEqual(pct_change.shape, (4,))
        np.testing.assert_allclose(annual, [10/3,5,2.5,5])
        self.assertFalse(falling.any())

    def test_region_mapping_rejects_unmapped_state(self):
        with self.assertRaises(ValueError):
            sol.create_region_mapping({'North':['A']}, ['A','B'])

    def test_region_mapping_rejects_duplicate_membership(self):
        with self.assertRaises(ValueError):
            sol.create_region_mapping({'North':['A'], 'South':['A']}, ['A'])

    def test_relative_income_index_uses_column_median(self):
        matrix = np.array([[10.,20.],[30.,40.]])
        actual = sol.calculate_relative_income_index(matrix)
        np.testing.assert_allclose(actual, [[0.5, 2/3],[1.5,4/3]])

    def test_poverty_change_shape_and_rising_count(self):
        poverty = np.array([[5.,6.,4.,3.],[7.,7.,8.,6.]])
        changes, rising = sol.calculate_poverty_change_2016_2019(poverty, sol.POVERTY_YEARS, ['A','B'])
        self.assertEqual(changes.shape, (2,3))
        self.assertEqual(rising, 1)

    def test_build_state_profile_has_expected_state_and_region(self):
        states, median, mean, poverty = sol.build_matrices(self.income, self.poverty)
        regions = {'Northern':['A','B']}
        profile = sol.build_state_profile('A', states, sol.SURVEY_YEARS, median, mean, poverty, sol.POVERTY_YEARS, regions)
        self.assertEqual(profile['state'], 'A')
        self.assertEqual(profile['region'], 'Northern')
        self.assertEqual(len(profile['median_incomes']), 5)
        self.assertEqual(len(profile['absolute_poverty']), 4)

    def test_build_state_profile_unknown_state_raises(self):
        states, median, mean, poverty = sol.build_matrices(self.income, self.poverty)
        with self.assertRaises(ValueError):
            sol.build_state_profile('Atlantis', states, sol.SURVEY_YEARS, median, mean, poverty, sol.POVERTY_YEARS, {'Northern':['A','B']})


class TestRealDatasetIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import json
        from pathlib import Path
        base = Path(__file__).resolve().parent / "data" / "raw"
        with (base / "hh_income_state.json").open(encoding="utf-8") as f:
            cls.income = json.load(f)["records"]
        with (base / "hh_poverty_state.json").open(encoding="utf-8") as f:
            cls.poverty = json.load(f)["records"]

    def test_raw_income_dataset_has_319_records(self):
        self.assertEqual(len(self.income), 319)

    def test_raw_poverty_dataset_has_310_records(self):
        self.assertEqual(len(self.poverty), 310)

    def test_real_data_builds_expected_matrix_shapes(self):
        states, median, mean, poverty = sol.build_matrices(self.income, self.poverty)
        self.assertEqual(len(states), 16)
        self.assertEqual(median.shape, (16, 5))
        self.assertEqual(mean.shape, (16, 5))
        self.assertEqual(poverty.shape, (16, 4))

    def test_real_data_has_no_missing_values_in_selected_years(self):
        _, median, mean, poverty = sol.build_matrices(self.income, self.poverty)
        self.assertEqual(int(np.isnan(median).sum()), 0)
        self.assertEqual(int(np.isnan(mean).sum()), 0)
        self.assertEqual(int(np.isnan(poverty).sum()), 0)

    def test_real_data_regions_cover_all_states(self):
        states, *_ = sol.build_matrices(self.income, self.poverty)
        mapping = sol.create_region_mapping(sol.REGIONS, states)
        self.assertEqual(set(mapping), set(states))


if __name__ == '__main__':
    unittest.main(verbosity=2)
