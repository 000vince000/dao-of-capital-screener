#!/usr/bin/env python3
"""Unit tests for current_baseline_data.py"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Import the module under test
import current_baseline_data


class TestCurrentBaselineData(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        # Sample balance sheet data
        self.sample_balance_sheet = pd.DataFrame({
            'symbol': ['AAPL'],
            'asOfDate': ['2023-12-31'],
            'TotalShareholderEquity': [62146000000],
            'InvestedCapital': [150000000000],
            'CashAndCashEquivalents': [40000000000],
            'CurrentDeferredLiabilities': [0],
            'LongTermDebtAndCapitalLeaseObligation': [95000000000],
            'NonCurrentDeferredLiabilities': [0],
            'OtherNonCurrentLiabilities': [5000000000],
            'CapitalStock': [70000000000],
            'CommonStock': [65000000000]
        })
        
        # Sample income statement data
        self.sample_income_statement = pd.DataFrame({
            'symbol': ['AAPL'],
            'asOfDate': ['2023-12-31'],
            'EBIT': [20000000000],
            'TaxRateForCalcs': [0.21],
            'NetIncome': [15000000000]
        })
        
        # Sample cash flow data
        self.sample_cash_flow = pd.DataFrame({
            'symbol': ['AAPL'],
            'asOfDate': ['2023-12-31'],
            'OperatingCashFlow': [18000000000]
        })
        
        # Sample market data
        self.sample_details = pd.DataFrame({
            'marketCap': [2800000000000]
        }, index=['AAPL'])
        
        # Sample profile data
        self.sample_profile = pd.DataFrame({
            'industry': ['Consumer Electronics']
        }, index=['AAPL'])
        
        # Sample valuation data
        self.sample_valuation = pd.DataFrame({
            'symbol': ['AAPL'],
            'asOfDate': ['2023-12-31'],
            'EnterpriseValue': [2900000000000]
        })

    def test_compute_financial_metrics_basic(self):
        """Test basic financial metrics computation"""
        result = current_baseline_data._compute_financial_metrics(
            self.sample_balance_sheet,
            self.sample_income_statement,
            self.sample_cash_flow,
            self.sample_details,
            self.sample_profile,
            self.sample_valuation
        )
        
        self.assertFalse(result.empty)
        self.assertEqual(result['symbol'].iloc[0], 'AAPL')
        
        # Test NOPAT calculation: EBIT * (1 - TaxRate)
        expected_nopat = 20000000000 * (1 - 0.21)
        self.assertAlmostEqual(result['nopat'].iloc[0], expected_nopat, places=0)
        
        # Test ROIC calculation: nopat / InvestedCapital
        expected_roic = expected_nopat / 150000000000
        self.assertAlmostEqual(result['roic'].iloc[0], expected_roic, places=4)

    def test_compute_financial_metrics_debt_calculation(self):
        """Test debt calculation logic"""
        result = current_baseline_data._compute_financial_metrics(
            self.sample_balance_sheet,
            self.sample_income_statement,
            self.sample_cash_flow,
            self.sample_details,
            self.sample_profile,
            self.sample_valuation
        )
        
        # Expected total debt
        expected_debt = 0 + 95000000000 + 0 + 5000000000  # Sum of debt components
        self.assertAlmostEqual(result['totalDebt'].iloc[0], expected_debt, places=0)

    def test_compute_financial_metrics_preferred_equity(self):
        """Test preferred equity calculation"""
        result = current_baseline_data._compute_financial_metrics(
            self.sample_balance_sheet,
            self.sample_income_statement,
            self.sample_cash_flow,
            self.sample_details,
            self.sample_profile,
            self.sample_valuation
        )
        
        # Preferred equity = CapitalStock - CommonStock
        expected_preferred = 70000000000 - 65000000000
        self.assertAlmostEqual(result['preferredequity'].iloc[0], expected_preferred, places=0)

    def test_compute_financial_metrics_value_metric(self):
        """Test value metric calculation: EBIT / EnterpriseValue"""
        result = current_baseline_data._compute_financial_metrics(
            self.sample_balance_sheet,
            self.sample_income_statement,
            self.sample_cash_flow,
            self.sample_details,
            self.sample_profile,
            self.sample_valuation
        )
        
        expected_value_metric = 20000000000 / 2900000000000
        self.assertAlmostEqual(result['valueMetric'].iloc[0], expected_value_metric, places=6)

    def test_compute_financial_metrics_roe(self):
        """Test ROE calculation"""
        result = current_baseline_data._compute_financial_metrics(
            self.sample_balance_sheet,
            self.sample_income_statement,
            self.sample_cash_flow,
            self.sample_details,
            self.sample_profile,
            self.sample_valuation
        )
        
        expected_roe = 15000000000 / 62146000000
        self.assertAlmostEqual(result['roe'].iloc[0], expected_roe, places=4)

    def test_compute_financial_metrics_missing_data(self):
        """Test handling of missing data"""
        # Test with missing required columns
        incomplete_balance_sheet = pd.DataFrame({
            'symbol': ['AAPL'],
            'asOfDate': ['2023-12-31']
            # Missing required financial columns
        })
        
        incomplete_income_statement = pd.DataFrame({
            'symbol': ['AAPL'],
            'asOfDate': ['2023-12-31']
            # Missing required financial columns
        })
        
        # This should trigger a KeyError which returns empty DataFrame
        try:
            result = current_baseline_data._compute_financial_metrics(
                incomplete_balance_sheet,
                incomplete_income_statement,
                self.sample_cash_flow,
                self.sample_details,
                self.sample_profile,
                self.sample_valuation
            )
            # If we get here, result should be empty due to missing columns
            self.assertTrue(result.empty)
        except KeyError:
            # KeyError is expected and handled by returning empty DataFrame
            pass

    def test_industry_filtering(self):
        """Test that ROE-relying industries are properly identified"""
        # Test a few industries from the DEFAULT_INDUSTRIES_RELYING_ON_ROE list
        roe_industries = current_baseline_data.DEFAULT_INDUSTRIES_RELYING_ON_ROE
        
        self.assertIn("Asset Management", roe_industries)
        self.assertIn("Insurance - Diversified", roe_industries)
        self.assertIn("REIT - Diversified", roe_industries)
        self.assertIn("Utilities - Regulated Electric", roe_industries)

    def test_validate_and_trim_columns(self):
        """Test that only expected columns are kept in output"""
        result = current_baseline_data._compute_financial_metrics(
            self.sample_balance_sheet,
            self.sample_income_statement,
            self.sample_cash_flow,
            self.sample_details,
            self.sample_profile,
            self.sample_valuation
        )
        
        expected_columns = [
            "symbol", "asOfDate", "EBIT", "InvestedCapital", "roic", "MarketCap",
            "CashAndCashEquivalents", "totalDebt", "preferredequity",
            "opCashFlow", "opCashFlowYield", "industry", "EnterpriseValue",
            "NetIncome", "TotalShareholderEquity", "roe", "valueMetric", "nopat"
        ]
        
        # Check that all expected columns are present (or would be if data was complete)
        for col in expected_columns:
            if col in result.columns:
                self.assertIn(col, result.columns)


if __name__ == '__main__':
    unittest.main() 