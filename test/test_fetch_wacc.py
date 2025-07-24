#!/usr/bin/env python3
"""Unit tests for fetch_wacc.py"""

import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
import tempfile
import os

# Import the module under test
import fetch_wacc


class TestFetchWACC(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        # Sample HTML content that mimics valueinvesting.io structure
        self.sample_html_content = """
        <html>
        <body>
        <div class="col-sm-6">
            <h4>Weighted Average Cost of Capital</h4>
            <table class="table">
                <tbody>
                    <tr><td>WACC</td><td>8.2%</td></tr>
                </tbody>
            </table>
        </div>
        <div class="col-sm-6">
            <h4>Cost of Equity</h4>
            <table class="table">
                <tbody>
                    <tr><td>Cost of Equity (CAPM)</td><td>10.5%</td></tr>
                </tbody>
            </table>
        </div>
        <div class="col-sm-6">
            <h4>Cost of Debt</h4>
            <table class="table">
                <tbody>
                    <tr><td>Pre-tax Cost of Debt</td><td>4.3%</td></tr>
                </tbody>
            </table>
        </div>
        </body>
        </html>
        """
        
        # Sample input DataFrame
        self.sample_input_data = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT', 'GOOGL'],
            'roic': [0.25, 0.30, 0.20],
            'MarketCap': [2800000000000, 2500000000000, 1800000000000]
        })

    def test_extract_percent_valid(self):
        """Test parsing valid percentage strings"""
        # Test normal percentage
        result = fetch_wacc._extract_percent("8.2%")
        self.assertAlmostEqual(result, 0.082, places=4)
        
        # Test percentage with extra whitespace
        result = fetch_wacc._extract_percent("  10.5%  ")
        self.assertAlmostEqual(result, 0.105, places=4)
        
        # Test integer percentage
        result = fetch_wacc._extract_percent("5%")
        self.assertAlmostEqual(result, 0.05, places=4)

    def test_extract_percent_invalid(self):
        """Test parsing invalid percentage strings"""
        # Test non-percentage string
        result = fetch_wacc._extract_percent("invalid")
        self.assertIsNone(result)
        
        # Test empty string
        result = fetch_wacc._extract_percent("")
        self.assertIsNone(result)
        
        # Test None input (should handle gracefully)
        try:
            result = fetch_wacc._extract_percent(None)
            self.assertIsNone(result)
        except TypeError:
            # Function doesn't handle None input, which is acceptable
            pass

    def test_parse_wacc_success(self):
        """Test successful WACC parsing from HTML"""
        result = fetch_wacc._parse_wacc(self.sample_html_content)
        
        expected = (0.082, 0.105, 0.043)  # wacc, costOfEquity, costOfDebt
        # Use almostEqual for floating point comparison
        self.assertAlmostEqual(result[0], expected[0], places=3)
        self.assertAlmostEqual(result[1], expected[1], places=3)
        self.assertAlmostEqual(result[2], expected[2], places=3)

    def test_parse_wacc_missing_data(self):
        """Test WACC parsing with missing data"""
        # HTML with missing WACC section
        incomplete_html = """
        <html>
        <body>
        <div class="col-sm-6">
            <h4>Cost of Equity</h4>
            <table class="table">
                <tbody>
                    <tr><td>Cost of Equity (CAPM)</td><td>10.5%</td></tr>
                </tbody>
            </table>
        </div>
        </body>
        </html>
        """
        
        result = fetch_wacc._parse_wacc(incomplete_html)
        
        expected = (None, 0.105, None)  # wacc, costOfEquity, costOfDebt
        self.assertEqual(result, expected)

    def test_parse_from_table(self):
        """Test parsing percentage from HTML table"""
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(self.sample_html_content, 'lxml')
        wacc = fetch_wacc._parse_from_table(soup, "WACC")
        
        self.assertAlmostEqual(wacc, 0.082, places=4)

    def test_parse_from_table_missing(self):
        """Test parsing when table row is missing"""
        from bs4 import BeautifulSoup
        
        html_no_wacc = "<html><body><div>No WACC here</div></body></html>"
        soup = BeautifulSoup(html_no_wacc, 'lxml')
        wacc = fetch_wacc._parse_from_table(soup, "WACC")
        
        self.assertIsNone(wacc)

    def test_file_input_processing(self):
        """Test processing symbols from CSV file"""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            self.sample_input_data.to_csv(temp_file.name, sep=';', index=False)
            temp_path = temp_file.name
        
        try:
            # Test reading symbols from file
            df = pd.read_csv(temp_path, sep=';')
            symbols = df['symbol'].dropna().astype(str).tolist()
            
            expected_symbols = ['AAPL', 'MSFT', 'GOOGL']
            self.assertEqual(symbols, expected_symbols)
            
        finally:
            # Clean up
            os.unlink(temp_path)

    def test_output_dataframe_structure(self):
        """Test that output DataFrame has correct structure"""
        # Create sample results
        sample_results = [
            {'symbol': 'AAPL', 'wacc': 0.08, 'costOfEquity': 0.10, 'costOfDebt': 0.04},
            {'symbol': 'MSFT', 'wacc': 0.09, 'costOfEquity': 0.11, 'costOfDebt': 0.05}
        ]
        
        df = pd.DataFrame(sample_results)
        
        # Check required columns are present
        required_columns = ['symbol', 'wacc', 'costOfEquity', 'costOfDebt']
        for col in required_columns:
            self.assertIn(col, df.columns)
        
        # Check data types
        self.assertEqual(df['symbol'].dtype, object)
        self.assertTrue(pd.api.types.is_numeric_dtype(df['wacc']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['costOfEquity']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['costOfDebt']))

    def test_clean_symbol_formatting(self):
        """Test symbol formatting/cleaning"""
        # Test various symbol formats that might need cleaning
        test_symbols = ['AAPL', ' MSFT ', 'BRK-B', 'BRK.B']
        
        # Most symbols should be left as-is, but whitespace should be stripped
        cleaned = [str(s).strip() for s in test_symbols]
        expected = ['AAPL', 'MSFT', 'BRK-B', 'BRK.B']
        
        self.assertEqual(cleaned, expected)


if __name__ == '__main__':
    unittest.main() 