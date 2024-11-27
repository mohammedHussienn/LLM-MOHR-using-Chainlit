import unittest
from unittest.mock import Mock, patch
import sys
import os
import pandas as pd
from datetime import datetime

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database_backend import DatabaseBackend

class TestDatabaseBackend(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.db = DatabaseBackend()
        
    def test_get_tenant_id(self):
        """Test get_tenant_id method with various inputs."""
        with patch.object(self.db.database, 'run') as mock_run:
            # Test valid tenant
            mock_run.return_value = "('3323',)"
            self.assertEqual(self.db.get_tenant_id('aladdin'), '3323')
            
            # Test non-existent tenant
            mock_run.return_value = "()"
            self.assertEqual(self.db.get_tenant_id('nonexistent'), '')
            
            # Test tenant with special characters
            mock_run.return_value = "('4455',)"
            self.assertEqual(self.db.get_tenant_id('test-tenant'), '4455')

    def test_get_prompt(self):
        """Test get_prompt method for all modes."""
        # Test valid modes
        self.assertIsNotNone(self.db.get_prompt('r'))
        self.assertIsNotNone(self.db.get_prompt('i'))
        self.assertIsNotNone(self.db.get_prompt('c'))
        
        # Test invalid mode
        with self.assertRaises(ValueError):
            self.db.get_prompt('invalid')

    def test_get_llm(self):
        """Test get_llm method for all modes."""
        # Test temperature settings for each mode
        llm_r = self.db.get_llm('r')
        llm_i = self.db.get_llm('i')
        llm_c = self.db.get_llm('c')
        
        self.assertEqual(llm_r.temperature, 0)
        self.assertEqual(llm_i.temperature, 0.4)
        self.assertEqual(llm_c.temperature, 0.9)

    def test_get_all_tenants(self):
        """Test get_all_tenants method."""
        with patch.object(self.db.database, 'run') as mock_run:
            # Test with multiple tenants
            mock_run.return_value = "[('Tenant1',), ('Tenant2',), ('Tenant3',)]"
            expected = ['Tenant1', 'Tenant2', 'Tenant3']
            self.assertEqual(self.db.get_all_tenants(), expected)
            
            # Test with single tenant
            mock_run.return_value = "[('SingleTenant',)]"
            self.assertEqual(self.db.get_all_tenants(), ['SingleTenant'])
            
            # Test with empty result
            mock_run.return_value = "[]"
            self.assertEqual(self.db.get_all_tenants(), [])

    def test_get_info_from_sql(self):
        """Test get_info_from_sql method."""
        with patch.object(self.db.database, 'run') as mock_run:
            # Test successful query
            mock_run.return_value = "[(1, 'Test')]"
            self.assertEqual(self.db.get_info_from_sql("SELECT * FROM test"), "[(1, 'Test')]")
            
            # Test empty result
            mock_run.return_value = "[]"
            self.assertEqual(self.db.get_info_from_sql("SELECT * FROM empty"), "[]")

    def test_create_csv(self):
        """Test create_csv method."""
        # Test mode 'r' with valid data
        test_input = {
            'results': "[(1, 'John Doe', '2024-01-01'), (2, 'Jane Smith', '2024-01-02')]",
            'column_names': ['ID', 'Name', 'Date']
        }
        df, message = self.db.create_csv('r', test_input)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (2, 3))
        self.assertEqual(list(df.columns), ['ID', 'Name', 'Date'])
        
        # Test with empty data
        empty_input = {
            'results': "[]",
            'column_names': ['ID', 'Name', 'Date']
        }
        df, message = self.db.create_csv('r', empty_input)
        self.assertIsNone(df)
        self.assertEqual(message, "No data found")
        
        # Test with invalid input
        invalid_input = "invalid data"
        df, message = self.db.create_csv('r', invalid_input)
        self.assertIsNone(df)
        self.assertEqual(message, "Invalid input type")

    @patch('database_backend.ChatOpenAI')
    def test_invoke_prompt(self, mock_chat):
        """Test invoke_prompt method."""
        # Create a mock for the entire SQLAgent
        mock_agent = Mock() 
        self.db.SQLAgent = mock_agent
        
        with patch.object(self.db, 'get_tenant_id') as mock_tenant_id, \
             patch.object(self.db, 'get_info_from_sql') as mock_sql:
            
            # Set up the mocks
            mock_tenant_id.return_value = '1234'
            mock_agent.return_value = {
                'output': '{"sql_query": "SELECT * FROM test", "column_names": ["ID", "Name"]}'
            }
            mock_sql.return_value = "[(1, 'Test')]"
            
            # Test raw mode ('r')
            result = self.db.invoke_prompt('r', "test query", "test_user")
            self.assertIsInstance(result, dict)
            self.assertIn('results', result)
            self.assertIn('column_names', result)
            
            # Verify the mocks were called correctly
            mock_tenant_id.assert_called_with('test_user')
            mock_agent.assert_called_once()
            
            # Test informative mode ('i')
            mock_chat.return_value.invoke.return_value = "Informative response"
            result = self.db.invoke_prompt('i', "test query", "test_user")
            self.assertIsInstance(result, str)
            
            # Test error handling
            mock_agent.side_effect = Exception("Test error")
            result = self.db.invoke_prompt('r', "test query", "test_user")
            self.assertTrue("Error executing query" in result)

if __name__ == '__main__':
    unittest.main()  