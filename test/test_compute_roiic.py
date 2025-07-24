#!/usr/bin/env python3
"""Unit tests for compute_roiic.py"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Import the module under test
import compute_roiic


class TestComputeROIIC(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        # Sample annual income statement data
        self.sample_income_annual = pd.DataFrame({
            'symbol': ['AAPL'] * 5,
            'asOfDate': ['2019-12-31', '2020-12-31', '2021-12-31', '2022-12-31', '2023-12-31'],
            'EBIT': [18000000000, 20000000000, 22000000000, 24000000000, 26000000000],
            'TaxRateForCalcs': [0.21, 0.21, 0.21, 0.21, 0.21]
        })
        
        # Sample annual balance sheet data
        self.sample_balance_annual = pd.DataFrame({
            'symbol': ['AAPL'] * 5,
            'asOfDate': ['2019-12-31', '2020-12-31', '2021-12-31', '2022-12-31', '2023-12-31'],
            'InvestedCapital': [100000000000, 110000000000, 120000000000, 130000000000, 140000000000]
        })
        
        # Sample baseline data with current year
        self.sample_baseline_data = pd.DataFrame({
            'symbol': ['AAPL'],
            'asOfDate': ['2023-12-31'],
            'nopat': [20540000000],  # 26000000000 * (1-0.21)
            'InvestedCapital': [140000000000]
        })

    def test_compute_nopat_and_invested_capital(self):
        """Test NOPAT and InvestedCapital computation for annual data"""
        result = compute_roiic.compute_nopat_and_invested_capital(
            self.sample_income_annual, 
            self.sample_balance_annual
        )
        
        self.assertFalse(result.empty)
        self.assertEqual(len(result), 5)  # Should have 5 years of data
        
        # Check NOPAT calculation for first row: 18000000000 * (1 - 0.21)
        expected_nopat_2019 = 18000000000 * (1 - 0.21)
        self.assertAlmostEqual(result.iloc[0]['nopat'], expected_nopat_2019, places=0)
        
        # Check that years are properly extracted
        self.assertIn('year', result.columns)
        self.assertEqual(result.iloc[0]['year'], 2019)
        self.assertEqual(result.iloc[-1]['year'], 2023)

    def test_compute_roiic_slope_sufficient_data(self):
        """Test ROIIC slope calculation with sufficient data points"""
        # Create test data with clear linear trends
        test_data = pd.DataFrame({
            'year': [2019, 2020, 2021, 2022, 2023],
            'nopat': [10000000000, 12000000000, 14000000000, 16000000000, 18000000000],  # +2B per year
            'InvestedCapital': [100000000000, 120000000000, 140000000000, 160000000000, 180000000000]  # +20B per year
        })
        
        roiic = compute_roiic.compute_roiic_slope(test_data)
        
        self.assertIsNotNone(roiic)
        # NOPAT slope = 2B, InvestedCapital slope = 20B, so ROIIC = 2/20 = 0.1
        expected_roiic = 2000000000 / 20000000000
        self.assertAlmostEqual(roiic, expected_roiic, places=3)

    def test_compute_roiic_slope_insufficient_data(self):
        """Test ROIIC slope calculation with insufficient data points"""
        # Only 3 data points (less than required 4)
        test_data = pd.DataFrame({
            'year': [2021, 2022, 2023],
            'nopat': [14000000000, 16000000000, 18000000000],
            'InvestedCapital': [140000000000, 160000000000, 180000000000]
        })
        
        roiic = compute_roiic.compute_roiic_slope(test_data)
        
        self.assertIsNone(roiic)

    def test_compute_roiic_slope_negative_ic_slope(self):
        """Test ROIIC slope calculation with negative InvestedCapital slope"""
        # InvestedCapital decreases over time
        test_data = pd.DataFrame({
            'year': [2019, 2020, 2021, 2022, 2023],
            'nopat': [10000000000, 12000000000, 14000000000, 16000000000, 18000000000],
            'InvestedCapital': [180000000000, 160000000000, 140000000000, 120000000000, 100000000000]  # Decreasing
        })
        
        roiic = compute_roiic.compute_roiic_slope(test_data)
        
        # Expect ROIIC = 2B / (-20B) = -0.1
        expected_roiic = -0.1
        self.assertAlmostEqual(roiic, expected_roiic, places=3)

    def test_compute_roiic_slope_zero_ic_slope(self):
        """Test ROIIC slope calculation with zero InvestedCapital slope"""
        # InvestedCapital stays constant
        test_data = pd.DataFrame({
            'year': [2019, 2020, 2021, 2022, 2023],
            'nopat': [10000000000, 12000000000, 14000000000, 16000000000, 18000000000],
            'InvestedCapital': [150000000000, 150000000000, 150000000000, 150000000000, 150000000000]  # Constant
        })
        
        roiic = compute_roiic.compute_roiic_slope(test_data)
        
        # Should return None for zero IC slope
        self.assertIsNone(roiic)

    def test_empty_data_handling(self):
        """Test handling of empty dataframes"""
        empty_income = pd.DataFrame()
        empty_balance = pd.DataFrame()
        
        result = compute_roiic.compute_nopat_and_invested_capital(empty_income, self.sample_balance_annual)
        self.assertTrue(result.empty)
        
        result = compute_roiic.compute_nopat_and_invested_capital(self.sample_income_annual, empty_balance)
        self.assertTrue(result.empty)

    @patch('compute_roiic.fetch_annual_data')
    def test_process_ticker_with_baseline_supplement(self, mock_fetch):
        """Test processing a ticker with baseline data supplementation"""
        # Mock the fetch_annual_data function
        mock_fetch.return_value = (self.sample_income_annual, self.sample_balance_annual)
        
        delay_ref = [1.0]
        
        # Test with baseline data
        symbol, roiic, data_points = compute_roiic.process_ticker(
            'AAPL', 
            delay_ref, 
            self.sample_baseline_data
        )
        
        self.assertEqual(symbol, 'AAPL')
        self.assertIsNotNone(roiic)
        self.assertGreaterEqual(data_points, 4)

    @patch('compute_roiic.fetch_annual_data')
    def test_process_ticker_no_baseline(self, mock_fetch):
        """Test processing a ticker without baseline data"""
        # Mock the fetch_annual_data function
        mock_fetch.return_value = (self.sample_income_annual, self.sample_balance_annual)
        
        delay_ref = [1.0]
        
        # Test without baseline data
        symbol, roiic, data_points = compute_roiic.process_ticker(
            'AAPL', 
            delay_ref, 
            None
        )
        
        self.assertEqual(symbol, 'AAPL')
        self.assertIsNotNone(roiic)
        self.assertEqual(data_points, 5)  # Should have 5 historical data points

    @patch('compute_roiic.fetch_annual_data')
    def test_process_ticker_no_data(self, mock_fetch):
        """Test processing a ticker with no available data"""
        # Mock the fetch_annual_data function to return empty dataframes
        mock_fetch.return_value = (pd.DataFrame(), pd.DataFrame())
        
        delay_ref = [1.0]
        
        symbol, roiic, data_points = compute_roiic.process_ticker(
            'INVALID', 
            delay_ref, 
            None
        )
        
        self.assertEqual(symbol, 'INVALID')
        self.assertIsNone(roiic)
        self.assertEqual(data_points, 0)

    def test_realistic_roiic_calculation(self):
        """Test ROIIC calculation with realistic data trends"""
        # Simulate a company with improving efficiency
        realistic_data = pd.DataFrame({
            'year': [2019, 2020, 2021, 2022, 2023],
            'nopat': [5000000000, 5500000000, 6200000000, 7000000000, 8000000000],  # Growing NOPAT
            'InvestedCapital': [50000000000, 55000000000, 58000000000, 60000000000, 62000000000]  # Slower capital growth
        })
        
        roiic = compute_roiic.compute_roiic_slope(realistic_data)
        
        self.assertIsNotNone(roiic)
        # This should represent a good ROIIC since NOPAT grows faster than InvestedCapital
        self.assertGreater(roiic, 0.1)  # Should be a decent return on incremental capital


if __name__ == '__main__':
    unittest.main() 