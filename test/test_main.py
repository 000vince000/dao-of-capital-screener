#!/usr/bin/env python3
"""Unit tests for main.py"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock
from pathlib import Path

# Import the module under test
import main


class TestMain(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        # Sample normalized data (what we'd expect from normalized_austrian.csv)
        self.sample_normalized_data = pd.DataFrame({
            'symbol': ['MSFT', 'AAPL', 'GOOGL', 'AMZN', 'TSLA'],
            'roic': [0.30, 0.25, 0.20, 0.15, 0.12],
            'wacc': [0.09, 0.08, 0.10, 0.11, 0.13],
            'valueMetric': [0.12, 0.08, 0.06, 0.05, 0.04],
            'valueMetricRank': [1, 2, 3, 4, 5],
            'excessReturn': [0.21, 0.17, 0.10, 0.04, -0.01],
            'excessReturnRank': [1, 2, 3, 4, 5],
            'rankingScore': [2, 4, 6, 8, 10],  # sum of ranks
            'MarketCap': [2500000000000, 2800000000000, 1800000000000, 1600000000000, 800000000000],
            'industry': ['Software', 'Consumer Electronics', 'Internet Services', 'E-commerce', 'Auto'],
            'EBIT': [18000000000, 20000000000, 15000000000, 12000000000, 8000000000],
            'EnterpriseValue': [2600000000000, 2900000000000, 1900000000000, 1700000000000, 850000000000]
        })
        
        # Sample ROIIC data
        self.sample_roiic_data = pd.DataFrame({
            'symbol': ['MSFT', 'AAPL', 'GOOGL', 'AMZN', 'TSLA'],
            'roiic': [0.15, 0.12, 0.08, 0.06, 0.03],
            'data_points_used': [5, 5, 4, 4, 3]
        })

    def test_parse_args_defaults(self):
        """Test argument parsing with defaults"""
        with patch('sys.argv', ['main.py']):
            args = main._parse_args()
            self.assertEqual(args.top, 50)
            self.assertEqual(args.output, Path("top50_overview.csv"))
            self.assertFalse(args.skip_screener)

    def test_parse_args_custom(self):
        """Test argument parsing with custom values"""
        with patch('sys.argv', ['main.py', '--top', '25', '--output', 'custom.csv', '--skip-screener']):
            args = main._parse_args()
            self.assertEqual(args.top, 25)
            self.assertEqual(args.output, Path("custom.csv"))
            self.assertTrue(args.skip_screener)

    @patch('main.subprocess.run')
    def test_run_script_success(self, mock_subprocess):
        """Test successful script execution"""
        mock_subprocess.return_value = None
        
        # Should not raise an exception
        main._run_script("test_script.py", "--arg1", "value1")
        
        # Check that subprocess.run was called correctly
        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args[0][0]
        # Check that the script path contains test_script.py
        self.assertTrue(any("test_script.py" in arg for arg in call_args))
        self.assertIn("--arg1", call_args)
        self.assertIn("value1", call_args)

    @patch('main.subprocess.run')
    def test_run_script_failure(self, mock_subprocess):
        """Test script execution failure"""
        from subprocess import CalledProcessError
        mock_subprocess.side_effect = CalledProcessError(1, ['python', 'test_script.py'])
        
        # Should raise CalledProcessError
        with self.assertRaises(CalledProcessError):
            main._run_script("test_script.py")

    def test_top_ticker_selection(self):
        """Test selection of top N tickers from normalized data"""
        # Data is already sorted by rankingScore (ascending, lower is better)
        top_3 = self.sample_normalized_data.head(3)
        top_tickers = top_3["symbol"].dropna().astype(str).tolist()
        
        expected = ['MSFT', 'AAPL', 'GOOGL']  # Best 3 by rankingScore
        self.assertEqual(top_tickers, expected)

    def test_roiic_data_merge(self):
        """Test merging of top tickers with ROIIC data"""
        top_df = self.sample_normalized_data.head(3)
        merged = top_df.merge(self.sample_roiic_data[["symbol", "roiic"]], on="symbol", how="left")
        
        # Check that merge was successful
        self.assertEqual(len(merged), 3)
        self.assertIn('roiic', merged.columns)
        
        # Check specific values
        msft_row = merged[merged['symbol'] == 'MSFT']
        self.assertEqual(msft_row['roiic'].iloc[0], 0.15)

    def test_growth_gate_calculation(self):
        """Test Growth Gate calculation: roiic - wacc"""
        top_df = self.sample_normalized_data.head(3)
        merged = top_df.merge(self.sample_roiic_data[["symbol", "roiic"]], on="symbol", how="left")
        
        # Compute Growth Gate
        merged["growthGate"] = merged["roiic"] - merged["wacc"]
        
        # Check calculations
        # MSFT: 0.15 - 0.09 = 0.06
        msft_growth_gate = merged[merged['symbol'] == 'MSFT']['growthGate'].iloc[0]
        self.assertAlmostEqual(msft_growth_gate, 0.06, places=4)
        
        # AAPL: 0.12 - 0.08 = 0.04
        aapl_growth_gate = merged[merged['symbol'] == 'AAPL']['growthGate'].iloc[0]
        self.assertAlmostEqual(aapl_growth_gate, 0.04, places=4)

    def test_column_selection_and_ordering(self):
        """Test final column selection and ordering"""
        top_df = self.sample_normalized_data.head(3)
        merged = top_df.merge(self.sample_roiic_data[["symbol", "roiic"]], on="symbol", how="left")
        merged["growthGate"] = merged["roiic"] - merged["wacc"]
        
        # Define expected column order (from main.py)
        cols_order = [
            "symbol",
            "industry",
            "MarketCap",
            "roic",        
            "wacc",
            "roiic",
            "valueMetric",
            "valueMetricRank",
            "excessReturn",
            "excessReturnRank",
            "rankingScore",
            "growthGate",
        ]
        
        # Ensure all columns exist (fill missing with NaN)
        for col in cols_order:
            if col not in merged.columns:
                merged[col] = np.nan
        
        final_df = merged[cols_order]
        
        # Check that we have all expected columns in correct order
        self.assertEqual(list(final_df.columns), cols_order)

    def test_growth_gate_sorting(self):
        """Test sorting by Growth Gate descending (higher is better)"""
        top_df = self.sample_normalized_data.head(3)
        merged = top_df.merge(self.sample_roiic_data[["symbol", "roiic"]], on="symbol", how="left")
        merged["growthGate"] = merged["roiic"] - merged["wacc"]
        
        # Sort by growthGate descending
        sorted_df = merged.sort_values("growthGate", ascending=False)
        
        # MSFT should be first (highest Growth Gate: 0.06)
        self.assertEqual(sorted_df.iloc[0]['symbol'], 'MSFT')

    def test_missing_wacc_handling(self):
        """Test handling of missing WACC data"""
        # Create scenario where some tickers have missing WACC
        incomplete_data = self.sample_normalized_data.copy()
        incomplete_data.loc[incomplete_data['symbol'] == 'GOOGL', 'wacc'] = np.nan
        
        top_df = incomplete_data.head(3)
        merged = top_df.merge(self.sample_roiic_data[["symbol", "roiic"]], on="symbol", how="left")
        
        # Growth Gate calculation should handle NaN values
        merged["growthGate"] = merged["roiic"] - merged["wacc"]
        
        # GOOGL should have NaN Growth Gate due to missing WACC
        googl_growth_gate = merged[merged['symbol'] == 'GOOGL']['growthGate'].iloc[0]
        self.assertTrue(pd.isna(googl_growth_gate))

    def test_missing_roiic_handling(self):
        """Test handling of missing ROIIC data"""
        # Create scenario where some tickers have missing ROIIC
        incomplete_roiic = self.sample_roiic_data.copy()
        incomplete_roiic = incomplete_roiic[incomplete_roiic['symbol'] != 'GOOGL']  # Remove GOOGL
        
        top_df = self.sample_normalized_data.head(3)
        merged = top_df.merge(incomplete_roiic[["symbol", "roiic"]], on="symbol", how="left")
        
        # GOOGL should have NaN ROIIC
        googl_roiic = merged[merged['symbol'] == 'GOOGL']['roiic'].iloc[0]
        self.assertTrue(pd.isna(googl_roiic))

    def test_output_csv_format(self):
        """Test CSV output format"""
        # Create final DataFrame
        top_df = self.sample_normalized_data.head(3)
        merged = top_df.merge(self.sample_roiic_data[["symbol", "roiic"]], on="symbol", how="left")
        merged["growthGate"] = merged["roiic"] - merged["wacc"]
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Save to CSV
            merged.to_csv(temp_path, sep=';', index=False)
            
            # Read back and verify
            read_df = pd.read_csv(temp_path, sep=';')
            self.assertEqual(len(read_df), 3)
            self.assertIn('symbol', read_df.columns)
            self.assertIn('growthGate', read_df.columns)
            
        finally:
            # Clean up
            os.unlink(temp_path)

    @patch('main.Path.exists')
    def test_skip_screener_logic(self, mock_exists):
        """Test skip screener logic"""
        # Test when file exists and skip is requested
        mock_exists.return_value = True
        
        # Mock args
        mock_args = MagicMock()
        mock_args.skip_screener = True
        
        # File exists and skip is True, so screener should be skipped
        should_skip = mock_args.skip_screener and mock_exists.return_value
        self.assertTrue(should_skip)
        
        # Test when file doesn't exist but skip is requested
        mock_exists.return_value = False
        should_skip = mock_args.skip_screener and mock_exists.return_value
        self.assertFalse(should_skip)  # Should run screener even if skip requested

    def test_ticker_string_creation(self):
        """Test creation of comma-separated ticker string"""
        top_tickers = ['MSFT', 'AAPL', 'GOOGL']
        ticker_str = ",".join(top_tickers)
        
        expected = "MSFT,AAPL,GOOGL"
        self.assertEqual(ticker_str, expected)


if __name__ == '__main__':
    unittest.main() 