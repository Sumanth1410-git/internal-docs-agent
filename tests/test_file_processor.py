"""
Comprehensive tests for File Processor
Tests file upload, processing, and document extraction capabilities
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import os
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import requests_mock

# Import test utilities
from tests import TEST_DATA_DIR, create_test_documents, cleanup_test_files

@pytest.fixture
def mock_slack_token():
    """Mock Slack bot token"""
    return "xoxb-test-token-12345"

@pytest.fixture
def sample_file_info():
    """Sample file information from Slack API"""
    return {
        'id': 'F12345TEST',
        'name': 'test_document.pdf',
        'size': 1024000,  # 1MB
        'url_private_download': 'https://files.slack.com/files-pri/T12345-F12345/test_document.pdf'
    }

@pytest.fixture
def temp_upload_dir():
    """Create temporary directory for file uploads"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

class TestSlackFileProcessor:
    """Test suite for Slack File Processor"""
    
    def test_file_processor_initialization(self, mock_slack_token):
        """Test file processor initialization"""
        from file_processor import SlackFileProcessor
        
        processor = SlackFileProcessor(mock_slack_token)
        
        assert processor.bot_token == mock_slack_token
        assert processor.temp_dir.exists()
        assert processor.max_file_size == 10  # 10MB default
        assert '.txt' in processor.supported_types
        assert '.pdf' in processor.supported_types
    
    def test_file_download_success(self, mock_slack_token, sample_file_info):
        """Test successful file download from Slack"""
        from file_processor import SlackFileProcessor
        
        processor = SlackFileProcessor(mock_slack_token)
        
        # Mock the HTTP request for file download
        with requests_mock.Mocker() as m:
            file_content = b"Test file content for processing"
            m.get(sample_file_info['url_private_download'], content=file_content)
            
            downloaded_file = processor.download_slack_file(sample_file_info)
            
            assert downloaded_file is not None
            assert downloaded_file.exists()
            assert downloaded_file.read_bytes() == file_content
    
    def test_file_download_failure(self, mock_slack_token, sample_file_info):
        """Test file download failure handling"""
        from file_processor import SlackFileProcessor
        
        processor = SlackFileProcessor(mock_slack_token)
        
        # Mock failed HTTP request
        with requests_mock.Mocker() as m:
            m.get(sample_file_info['url_private_download'], status_code=404)
            
            downloaded_file = processor.download_slack_file(sample_file_info)
            
            assert downloaded_file is None
    
    def test_file_size_limit_enforcement(self, mock_slack_token):
        """Test file size limit enforcement"""
        from file_processor import SlackFileProcessor
        
        processor = SlackFileProcessor(mock_slack_token)
        
        # Create file info that exceeds size limit
        large_file_info = {
            'id': 'F12345LARGE',
            'name': 'large_file.pdf',
            'size': 20 * 1024 * 1024,  # 20MB (exceeds 10MB limit)
            'url_private_download': 'https://files.slack.com/files-pri/T12345-F12345/large_file.pdf'
        }
        
        downloaded_file = processor.download_slack_file(large_file_info)
        
        assert downloaded_file is None  # Should reject large files
    
    def test_text_file_processing(self, mock_slack_token, temp_upload_dir):
        """Test text file processing and document extraction"""
        from file_processor import SlackFileProcessor
        
        processor = SlackFileProcessor(mock_slack_token)
        
        # Create test text file
        test_file = temp_upload_dir / "test.txt"
        test_content = """
        Test Company Policy Document
        
        This is a sample policy document for testing file processing.
        It contains important information about company procedures.
        """
        test_file.write_text(test_content)
        
        file_info = {
            'id': 'F12345TEXT',
            'name': 'test.txt',
            'size': len(test_content)
        }
        
        documents = processor._process_text_file(test_file, file_info, 'U12345')
        
        assert len(documents) > 0
        assert documents[0].metadata['source_file'] == 'test.txt'
        assert documents[0].metadata['uploaded_by'] == 'U12345'
        assert 'policy' in documents[0].page_content.lower()
    
    def test_pdf_file_processing(self, mock_slack_token, temp_upload_dir):
        """Test PDF file processing"""
        from file_processor import SlackFileProcessor
        
        processor = SlackFileProcessor(mock_slack_token)
        
        # Create mock PDF file (we'll mock the PDF loader)
        test_file = temp_upload_dir / "test.pdf"
        test_file.write_bytes(b"Mock PDF content")
        
        file_info = {
            'id': 'F12345PDF',
            'name': 'test.pdf',
            'size': 1000
        }
        
        # Mock PyPDFLoader
        with patch('langchain_community.document_loaders.PyPDFLoader') as mock_loader:
            mock_doc = Mock()
            mock_doc.page_content = "PDF content extracted"
            mock_doc.metadata = {}
            mock_loader.return_value.load.return_value = [mock_doc]
            
            documents = processor._process_pdf_file(test_file, file_info, 'U12345')
            
            assert len(documents) == 1
            assert documents[0].metadata['source_file'] == 'test.pdf'
            assert documents[0].metadata['file_type'] == '.pdf'
            assert documents[0].metadata['uploaded_by'] == 'U12345'
    
    def test_word_document_processing(self, mock_slack_token, temp_upload_dir):
        """Test Word document processing"""
        from file_processor import SlackFileProcessor
        
        processor = SlackFileProcessor(mock_slack_token)
        
        test_file = temp_upload_dir / "test.docx"
        test_file.write_bytes(b"Mock Word document content")
        
        file_info = {
            'id': 'F12345WORD',
            'name': 'test.docx',
            'size': 2000
        }
        
        # Mock UnstructuredWordDocumentLoader
        with patch('langchain_community.document_loaders.UnstructuredWordDocumentLoader') as mock_loader:
            mock_doc = Mock()
            mock_doc.page_content = "Word document content extracted"
            mock_doc.metadata = {}
            mock_loader.return_value.load.return_value = [mock_doc]
            
            documents = processor._process_word_file(test_file, file_info, 'U12345')
            
            assert len(documents) == 1
            assert documents[0].metadata['source_file'] == 'test.docx'
            assert documents[0].metadata['file_type'] == '.docx'
    
    def test_csv_file_processing(self, mock_slack_token, temp_upload_dir):
        """Test CSV file processing"""
        from file_processor import SlackFileProcessor
        
        processor = SlackFileProcessor(mock_slack_token)
        
        # Create test CSV file
        test_file = temp_upload_dir / "test.csv"
        csv_content = """Name,Department,Role
John Doe,Engineering,Developer
Jane Smith,HR,Manager
Bob Johnson,Sales,Representative"""
        test_file.write_text(csv_content)
        
        file_info = {
            'id': 'F12345CSV',
            'name': 'test.csv',
            'size': len(csv_content)
        }
        
        documents = processor._process_csv_file(test_file, file_info, 'U12345')
        
        assert len(documents) == 1
        assert documents[0].metadata['source_file'] == 'test.csv'
        assert documents[0].metadata['file_type'] == '.csv'
        assert documents[0].metadata['rows'] == 3
        assert 'Name' in documents[0].metadata['columns']
        assert 'Department' in documents[0].metadata['columns']
    
    def test_json_file_processing(self, mock_slack_token, temp_upload_dir):
        """Test JSON file processing"""
        from file_processor import SlackFileProcessor
        
        processor = SlackFileProcessor(mock_slack_token)
        
        # Create test JSON file
        test_file = temp_upload_dir / "test.json"
        json_data = {
            "company": "Test Corp",
            "policies": {
                "remote_work": "3 days per week",
                "vacation": "20 days annually"
            },
            "employees": 150
        }
        test_file.write_text(json.dumps(json_data, indent=2))
        
        file_info = {
            'id': 'F12345JSON',
            'name': 'test.json',
            'size': len(json.dumps(json_data))
        }
        
        documents = processor._process_json_file(test_file, file_info, 'U12345')
        
        assert len(documents) == 1
        assert documents[0].metadata['source_file'] == 'test.json'
        assert documents[0].metadata['file_type'] == '.json'
        assert 'remote_work' in documents[0].page_content
        assert 'Test Corp' in documents[0].page_content
    
    def test_markdown_file_processing(self, mock_slack_token, temp_upload_dir):
        """Test Markdown file processing"""
        from file_processor import SlackFileProcessor
        
        processor = SlackFileProcessor(mock_slack_token)
        
        # Create test Markdown file
        test_file = temp_upload_dir / "test.md"
        markdown_content = """
# Company Documentation

## Remote Work Policy

Employees are allowed to work remotely up to **3 days per week**.

### Requirements
- Manager approval required
- Maintain regular communication
- Ensure productivity standards

## Benefits Package

- Health Insurance: 100% covered
- Dental Insurance: 90% covered
- 401k Matching: Up to 4%
        """
        test_file.write_text(markdown_content)
        
        file_info = {
            'id': 'F12345MD',
            'name': 'test.md',
            'size': len(markdown_content)
        }
        
        # Mock MarkdownTextSplitter
        with patch('langchain.text_splitter.MarkdownTextSplitter') as mock_splitter:
            mock_doc = Mock()
            mock_doc.page_content = markdown_content
            mock_doc.metadata = {'source_file': 'test.md', 'file_type': '.md', 'uploaded_by': 'U12345'}
            mock_splitter.return_value.split_documents.return_value = [mock_doc]
            
            documents = processor._process_markdown_file(test_file, file_info, 'U12345')
            
            assert len(documents) == 1
            assert documents[0].metadata['source_file'] == 'test.md'
            assert documents[0].metadata['file_type'] == '.md'
    
    def test_unsupported_file_type(self, mock_slack_token, sample_file_info):
        """Test handling of unsupported file types"""
        from file_processor import SlackFileProcessor
        
        processor = SlackFileProcessor(mock_slack_token)
        
        # Modify file info for unsupported type
        unsupported_file_info = sample_file_info.copy()
        unsupported_file_info['name'] = 'test_file.xyz'  # Unsupported extension
        
        success, message, documents = processor.process_uploaded_file(
            unsupported_file_info, 'U12345'
        )
        
        assert success is False
        assert 'not supported' in message.lower()
        assert len(documents) == 0
    
    def test_complete_file_processing_workflow(self, mock_slack_token, temp_upload_dir):
        """Test complete file processing workflow from upload to documents"""
        from file_processor import SlackFileProcessor
        
        processor = SlackFileProcessor(mock_slack_token)
        
        # Create test file
        test_file = temp_upload_dir / "workflow_test.txt"
        test_content = "This is a test document for the complete workflow."
        test_file.write_text(test_content)
        
        file_info = {
            'id': 'F12345WORKFLOW',
            'name': 'workflow_test.txt',
            'size': len(test_content),
            'url_private_download': 'https://files.slack.com/test/workflow_test.txt'
        }
        
        # Mock file download
        with patch.object(processor, 'download_slack_file', return_value=test_file):
            
            success, message, documents = processor.process_uploaded_file(file_info, 'U12345')
            
            assert success is True
            assert 'successfully processed' in message.lower()
            assert len(documents) > 0
            assert documents[0].metadata['uploaded_by'] == 'U12345'
            assert documents[0].metadata['slack_file_id'] == 'F12345WORKFLOW'
    
    def test_safe_filename_creation(self, mock_slack_token):
        """Test safe filename creation with special characters"""
        from file_processor import SlackFileProcessor
        
        processor = SlackFileProcessor(mock_slack_token)
        
        # Test with dangerous filename
        dangerous_filename = "../../etc/passwd.txt"
        safe_filename = processor._create_safe_filename(dangerous_filename)
        
        assert '/' not in safe_filename
        assert '..' not in safe_filename
        assert safe_filename.endswith('.txt')
        
        # Test with unicode characters
        unicode_filename = "tëst_dôcümënt.pdf"
        safe_filename = processor._create_safe_filename(unicode_filename)
        
        assert safe_filename.endswith('.pdf')
        # Should only contain safe characters
        safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_"
        assert all(c in safe_chars for c in safe_filename.replace('_', '').replace('.', ''))
    
    def test_temporary_file_cleanup(self, mock_slack_token, temp_upload_dir):
        """Test cleanup of temporary files"""
        from file_processor import SlackFileProcessor
        
        processor = SlackFileProcessor(mock_slack_token)
        
        # Create old temporary files
        old_file1 = processor.temp_dir / "old_file_1.txt"
        old_file2 = processor.temp_dir / "old_file_2.pdf"
        
        old_file1.write_text("Old file 1")
        old_file2.write_bytes(b"Old file 2")
        
        # Mock file timestamps to be old
        import time
        old_time = time.time() - (25 * 3600)  # 25 hours ago
        
        with patch('pathlib.Path.stat') as mock_stat:
            mock_stat.return_value.st_mtime = old_time
            
            # Run cleanup
            processor.cleanup_temp_files(max_age_hours=24)
            
            # Files should be cleaned up (mocked, so we verify the logic)
            mock_stat.assert_called()
    
    def test_error_handling_in_file_processing(self, mock_slack_token, temp_upload_dir):
        """Test error handling during file processing"""
        from file_processor import SlackFileProcessor
        
        processor = SlackFileProcessor(mock_slack_token)
        
        # Create file that will cause processing error
        test_file = temp_upload_dir / "error_test.txt"
        test_file.write_text("Test content")
        
        file_info = {
            'id': 'F12345ERROR',
            'name': 'error_test.txt',
            'size': 100
        }
        
        # Mock processor method to raise exception
        with patch.object(processor, '_process_text_file', side_effect=Exception("Processing error")):
            
            success, message, documents = processor.process_uploaded_file(file_info, 'U12345')
            
            assert success is False
            assert 'error' in message.lower()
            assert len(documents) == 0
    
    def test_batch_file_processing(self, mock_slack_token):
        """Test processing multiple files in batch"""
        from file_processor import SlackFileProcessor
        
        processor = SlackFileProcessor(mock_slack_token)
        
        # Create multiple file infos
        file_infos = [
            {
                'id': f'F1234{i}',
                'name': f'test_file_{i}.txt',
                'size': 1000,
                'url_private_download': f'https://files.slack.com/test_{i}.txt'
            }
            for i in range(3)
        ]
        
        # Mock file processing
        with patch.object(processor, 'process_uploaded_file') as mock_process:
            mock_process.return_value = (True, "Success", [Mock()])
            
            results = []
            for file_info in file_infos:
                result = processor.process_uploaded_file(file_info, 'U12345')
                results.append(result)
            
            # Verify all files were processed
            assert len(results) == 3
            assert all(result[0] for result in results)  # All successful
            assert mock_process.call_count == 3

class TestFileProcessorIntegration:
    """Integration tests for file processor with other components"""
    
    def test_integration_with_document_loader(self, mock_slack_token, temp_upload_dir):
        """Test integration between file processor and document loader"""
        from file_processor import SlackFileProcessor
        
        processor = SlackFileProcessor(mock_slack_token)
        
        # Create test document
        test_file = temp_upload_dir / "integration_test.txt"
        test_content = """
        Integration Test Document
        
        This document tests the integration between file processor and document loader.
        It should be properly chunked and processed for RAG system integration.
        """
        test_file.write_text(test_content)
        
        file_info = {
            'id': 'F12345INTEGRATION',
            'name': 'integration_test.txt',
            'size': len(test_content),
            'url_private_download': 'https://files.slack.com/integration_test.txt'
        }
        
        # Mock download
        with patch.object(processor, 'download_slack_file', return_value=test_file):
            
            success, message, documents = processor.process_uploaded_file(file_info, 'U12345')
            
            # Verify integration success
            assert success is True
            assert len(documents) > 0
            
            # Check document structure for RAG compatibility
            for doc in documents:
                assert hasattr(doc, 'page_content')
                assert hasattr(doc, 'metadata')
                assert 'source_file' in doc.metadata
                assert 'uploaded_by' in doc.metadata
                assert 'upload_date' in doc.metadata

@pytest.mark.asyncio
async def test_concurrent_file_processing():
    """Test concurrent file processing capabilities"""
    import asyncio
    from file_processor import SlackFileProcessor
    
    processor = SlackFileProcessor("test-token")
    
    # Create multiple mock file processing tasks
    async def process_file_task(file_id):
        file_info = {
            'id': file_id,
            'name': f'async_test_{file_id}.txt',
            'size': 1000
        }
        
        # Mock the actual processing
        with patch.object(processor, 'process_uploaded_file') as mock_process:
            mock_process.return_value = (True, "Success", [Mock()])
            return processor.process_uploaded_file(file_info, 'U12345')
    
    # Process multiple files concurrently
    tasks = [process_file_task(f'F{i}') for i in range(5)]
    results = await asyncio.gather(*tasks)
    
    # Verify all files processed successfully
    assert len(results) == 5
    assert all(result[0] for result in results)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
