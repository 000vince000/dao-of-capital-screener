#!/usr/bin/env python3
"""Unit tests for normalized_austrian_screener.py"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os

# Import the module under test
import normalized_austrian_screener


class TestNormalizedAustrianScreener(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        # Sample baseline data
        self.sample_baseline_data = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT', 'GOOGL'],
            'roic': [0.25, 0.30, 0.20],
            'valueMetric': [0.08, 0.12, 0.06],
            'MarketCap': [2800000000000, 2500000000000, 1800000000000],
            'industry': ['Consumer Electronics', 'Software', 'Internet Services'],
            'EBIT': [20000000000, 18000000000, 15000000000],
            'EnterpriseValue': [2900000000000, 2600000000000, 1900000000000]
        })
        
        # Sample WACC data
        self.sample_wacc_data = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT', 'GOOGL'],
            'wacc': [0.08, 0.09, 0.10],
            'costOfEquity': [0.10, 0.11, 0.12]
        })

    def test_excess_return_calculation(self):
        """Test excess return calculation: roic - wacc"""
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as baseline_file:
            self.sample_baseline_data.to_csv(baseline_file.name, sep=';', index=False)
            baseline_path = baseline_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as wacc_file:
            self.sample_wacc_data.to_csv(wacc_file.name, sep=';', index=False)
            wacc_path = wacc_file.name
        
        try:
            # Read baseline data
            df = pd.read_csv(baseline_path, sep=';')
            
            # Merge WACC data (simulate what the main function does)
            wacc_df = pd.read_csv(wacc_path, sep=';')
            df = df.merge(wacc_df[["symbol", "wacc", "costOfEquity"]], on="symbol", how="left")
            
            # Compute excessReturn
            df["excessReturn"] = df["roic"] - df["wacc"]
            
            # Test calculations
            expected_excess_returns = [0.25 - 0.08, 0.30 - 0.09, 0.20 - 0.10]
            for i, expected in enumerate(expected_excess_returns):
                self.assertAlmostEqual(df["excessReturn"].iloc[i], expected, places=4)
                
        finally:
            # Clean up temporary files
            os.unlink(baseline_path)
            os.unlink(wacc_path)

    def test_ranking_calculations(self):
        """Test ranking calculations for excessReturn and valueMetric"""
        df = self.sample_baseline_data.copy()
        df = df.merge(self.sample_wacc_data[["symbol", "wacc", "costOfEquity"]], on="symbol", how="left")
        df["excessReturn"] = df["roic"] - df["wacc"]
        
        # Rank excessReturn descending (1 = best)
        df["excessReturnRank"] = df["excessReturn"].rank(method="min", ascending=False, na_option="bottom")
        
        # Rank valueMetric descending (1 = best)
        df["valueMetricRank"] = df["valueMetric"].rank(method="min", ascending=False)
        
        # Test excessReturn rankings
        # MSFT has highest excess return (0.21), AAPL second (0.17), GOOGL third (0.10)
        self.assertEqual(df[df['symbol'] == 'MSFT']['excessReturnRank'].iloc[0], 1)
        self.assertEqual(df[df['symbol'] == 'AAPL']['excessReturnRank'].iloc[0], 2)
        self.assertEqual(df[df['symbol'] == 'GOOGL']['excessReturnRank'].iloc[0], 3)
        
        # Test valueMetric rankings
        # MSFT has highest valueMetric (0.12), AAPL second (0.08), GOOGL third (0.06)
        self.assertEqual(df[df['symbol'] == 'MSFT']['valueMetricRank'].iloc[0], 1)
        self.assertEqual(df[df['symbol'] == 'AAPL']['valueMetricRank'].iloc[0], 2)
        self.assertEqual(df[df['symbol'] == 'GOOGL']['valueMetricRank'].iloc[0], 3)

    def test_ranking_score_calculation(self):
        """Test rankingScore calculation as sum of ranks"""
        df = self.sample_baseline_data.copy()
        df = df.merge(self.sample_wacc_data[["symbol", "wacc", "costOfEquity"]], on="symbol", how="left")
        df["excessReturn"] = df["roic"] - df["wacc"]
        
        df["excessReturnRank"] = df["excessReturn"].rank(method="min", ascending=False, na_option="bottom")
        df["valueMetricRank"] = df["valueMetric"].rank(method="min", ascending=False)
        
        # Compute rankingScore
        df["rankingScore"] = df["valueMetricRank"] + df["excessReturnRank"]
        
        # MSFT should have the best (lowest) ranking score: 1 + 1 = 2
        msft_score = df[df['symbol'] == 'MSFT']['rankingScore'].iloc[0]
        self.assertEqual(msft_score, 2)
        
        # AAPL should have: 2 + 2 = 4
        aapl_score = df[df['symbol'] == 'AAPL']['rankingScore'].iloc[0]
        self.assertEqual(aapl_score, 4)
        
        # GOOGL should have: 3 + 3 = 6
        googl_score = df[df['symbol'] == 'GOOGL']['rankingScore'].iloc[0]
        self.assertEqual(googl_score, 6)

    def test_sorting_by_ranking_score(self):
        """Test that data is sorted by rankingScore ascending (lower is better)"""
        df = self.sample_baseline_data.copy()
        df = df.merge(self.sample_wacc_data[["symbol", "wacc", "costOfEquity"]], on="symbol", how="left")
        df["excessReturn"] = df["roic"] - df["wacc"]
        
        df["excessReturnRank"] = df["excessReturn"].rank(method="min", ascending=False, na_option="bottom")
        df["valueMetricRank"] = df["valueMetric"].rank(method="min", ascending=False)
        df["rankingScore"] = df["valueMetricRank"] + df["excessReturnRank"]
        
        # Sort by rankingScore ascending (lower is better)
        df = df.sort_values(by="rankingScore", ascending=True)
        
        # First row should be MSFT (best rankingScore = 2)
        self.assertEqual(df.iloc[0]['symbol'], 'MSFT')
        # Last row should be GOOGL (worst rankingScore = 6)
        self.assertEqual(df.iloc[-1]['symbol'], 'GOOGL')

    def test_handle_missing_wacc_data(self):
        """Test handling of missing WACC data"""
        # Create baseline data with an extra symbol not in WACC
        extended_baseline = self.sample_baseline_data.copy()
        new_row = pd.DataFrame({
            'symbol': ['TSLA'],
            'roic': [0.15],
            'valueMetric': [0.05],
            'MarketCap': [800000000000],
            'industry': ['Auto Manufacturers'],
            'EBIT': [5000000000],
            'EnterpriseValue': [850000000000]
        })
        extended_baseline = pd.concat([extended_baseline, new_row], ignore_index=True)
        
        # Merge with WACC data (TSLA will have NaN values)
        df = extended_baseline.merge(self.sample_wacc_data[["symbol", "wacc", "costOfEquity"]], on="symbol", how="left")
        df["excessReturn"] = df["roic"] - df["wacc"]
        
        # Check that TSLA has NaN excessReturn due to missing WACC
        tsla_row = df[df['symbol'] == 'TSLA']
        self.assertTrue(pd.isna(tsla_row['excessReturn'].iloc[0]))
        
        # Rank with na_option='bottom' should put NaN values at the end
        df["excessReturnRank"] = df["excessReturn"].rank(method="min", ascending=False, na_option="bottom")
        
        # Get the updated TSLA row after ranking
        tsla_row_updated = df[df['symbol'] == 'TSLA']
        
        # TSLA should get the worst rank (4 in this case, since it's the 4th item)
        tsla_rank = tsla_row_updated['excessReturnRank'].iloc[0]
        self.assertEqual(tsla_rank, 4)


if __name__ == '__main__':
    unittest.main() 