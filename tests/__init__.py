"""
Test package for Internal Docs Q&A Agent
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
import os
import tempfile
from pathlib import Path

# Test configuration
TEST_DATA_DIR = Path(__file__).parent / "test_data"
TEST_DATA_DIR.mkdir(exist_ok=True)

def create_test_documents():
    """Create test documents for testing"""
    test_files = {
        "test_policy.txt": """
# Test Company Policy

## Remote Work Policy
Employees can work remotely up to 3 days per week.

## Refund Policy  
Full refunds within 30 days of purchase.
        """,
        
        "test_hr.txt": """
# HR Test Document

## Benefits
- Health insurance: 100% covered
- PTO: 20 days annually
- Sick leave: 10 days annually
        """,
        
        "test_technical.txt": """
# Technical Documentation

## API Endpoints
- Production: https://api.company.com
- Staging: https://staging-api.company.com
        """
    }
    
    for filename, content in test_files.items():
        file_path = TEST_DATA_DIR / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content.strip())
    
    return list(test_files.keys())

def cleanup_test_files():
    """Clean up test files"""
    import shutil
    if TEST_DATA_DIR.exists():
        shutil.rmtree(TEST_DATA_DIR)
        TEST_DATA_DIR.mkdir(exist_ok=True)
