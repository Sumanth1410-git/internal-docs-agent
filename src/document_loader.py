import os
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

from langchain_community.document_loaders import (
    TextLoader, 
    PyPDFLoader, 
    UnstructuredWordDocumentLoader,
    DirectoryLoader
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedDocumentLoader:
    """
    Document loader optimized for HP Victus i3 + 3040 GPU
    Handles multiple file types with memory-efficient processing
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, batch_size: int = 5):
        """
        Initialize with hardware-optimized settings
        
        Args:
            chunk_size: Size of text chunks (reduced for i3 efficiency)
            chunk_overlap: Overlap between chunks  
            batch_size: Number of documents to process simultaneously
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Supported file types
        self.supported_extensions = {
            '.txt': TextLoader,
            '.pdf': PyPDFLoader,
            '.docx': UnstructuredWordDocumentLoader,
            '.doc': UnstructuredWordDocumentLoader
        }
        
        logger.info(f"DocumentLoader initialized with chunk_size={chunk_size}, batch_size={batch_size}")
    
    def load_single_document(self, file_path: str) -> List[Document]:
        """Load a single document and return processed chunks"""
        try:
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension not in self.supported_extensions:
                logger.warning(f"Unsupported file type: {file_extension}")
                return []
            
            # Select appropriate loader
            loader_class = self.supported_extensions[file_extension]
            loader = loader_class(file_path)
            
            # Load document
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} pages from {Path(file_path).name}")
            
            # Split into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Add metadata
            for chunk in chunks:
                chunk.metadata.update({
                    'source_file': Path(file_path).name,
                    'file_type': file_extension,
                    'chunk_size': len(chunk.page_content)
                })
            
            logger.info(f"Created {len(chunks)} chunks from {Path(file_path).name}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            return []
    
    def load_directory(self, directory_path: str) -> List[Document]:
        """Load all supported documents from a directory"""
        if not os.path.exists(directory_path):
            logger.error(f"Directory not found: {directory_path}")
            return []
        
        all_chunks = []
        files = []
        
        # Find all supported files
        for ext in self.supported_extensions.keys():
            pattern = f"**/*{ext}"
            files.extend(Path(directory_path).glob(pattern))
        
        logger.info(f"Found {len(files)} supported files in {directory_path}")
        
        # Process files in batches (memory optimization)
        for i in range(0, len(files), self.batch_size):
            batch_files = files[i:i + self.batch_size]
            logger.info(f"Processing batch {i//self.batch_size + 1} ({len(batch_files)} files)")
            
            batch_chunks = []
            for file_path in batch_files:
                chunks = self.load_single_document(str(file_path))
                batch_chunks.extend(chunks)
            
            all_chunks.extend(batch_chunks)
            
            # Small delay to prevent system overload
            time.sleep(0.1)
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks
    
    def create_sample_documents(self, data_dir: str = "data") -> None:
        """Create sample documents for testing"""
        os.makedirs(data_dir, exist_ok=True)
        
        sample_docs = {
            "company_policies.txt": """
# Company Policies and Procedures

## Remote Work Policy
Employees can work remotely up to 3 days per week. Remote work requests must be approved by your direct manager.

## Refund Policy
We offer full refunds within 30 days of purchase. Refunds take 5-7 business days to process.

## Expense Reimbursement
Submit expense reports through the company portal within 30 days. Include receipts for all expenses over $25.

## Design Asset Requests
Contact the design team through Slack #design-requests channel. Include project details and deadline.

## IT Support
For technical issues, create a ticket at help.company.com or email support@company.com.
Response time: 24-48 hours for non-urgent issues.
            """,
            
            "hr_handbook.txt": """
# HR Handbook

## Employee Benefits
- Health insurance: 100% covered for employees, 80% for family
- Dental and vision: 90% covered
- 401k matching: Up to 4% of salary
- Paid time off: 20 days annually
- Sick leave: 10 days annually

## Performance Reviews
Conducted bi-annually in June and December. Self-assessments due one week before review meetings.

## Code of Conduct
Maintain professional behavior at all times. Report any issues to HR immediately.

## Training and Development
$2000 annual budget for professional development. Submit requests through the learning management system.
            """,
            
            "technical_docs.txt": """
# Technical Documentation

## API Access
Production API: https://api.company.com
Staging API: https://staging-api.company.com
Authentication: Bearer token required

## Database Backup
Daily backups at 2 AM EST. Retention period: 30 days for daily, 12 months for monthly.

## Deployment Process
1. Create pull request
2. Code review (minimum 2 approvals)
3. Merge to staging
4. QA testing
5. Deploy to production

## Security Guidelines
- Use 2FA for all accounts
- Rotate API keys quarterly
- Never commit secrets to version control
- Report security issues immediately

## Monitoring
System health dashboard: https://status.company.com
Alert notifications sent to #alerts Slack channel.
            """
        }
        
        for filename, content in sample_docs.items():
            file_path = os.path.join(data_dir, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content.strip())
            logger.info(f"Created sample document: {filename}")
    
    def get_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """Get statistics about processed documents"""
        if not documents:
            return {}
        
        stats = {
            'total_chunks': len(documents),
            'total_characters': sum(len(doc.page_content) for doc in documents),
            'average_chunk_size': sum(len(doc.page_content) for doc in documents) / len(documents),
            'file_types': {},
            'source_files': set()
        }
        
        for doc in documents:
            file_type = doc.metadata.get('file_type', 'unknown')
            source_file = doc.metadata.get('source_file', 'unknown')
            
            stats['file_types'][file_type] = stats['file_types'].get(file_type, 0) + 1
            stats['source_files'].add(source_file)
        
        stats['unique_files'] = len(stats['source_files'])
        stats['source_files'] = list(stats['source_files'])
        
        return stats

# Test function
def test_document_loader():
    """Test the document loader functionality"""
    print("=== Testing Document Loader ===")
    
    # Initialize loader
    loader = OptimizedDocumentLoader(chunk_size=500, batch_size=5)
    
    # Create sample documents
    print("Creating sample documents...")
    loader.create_sample_documents()
    
    # Load documents
    print("Loading documents from data directory...")
    documents = loader.load_directory("data")
    
    # Display statistics
    stats = loader.get_stats(documents)
    print(f"\n=== Document Loading Statistics ===")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Total characters: {stats['total_characters']:,}")
    print(f"Average chunk size: {stats['average_chunk_size']:.1f}")
    print(f"Unique files: {stats['unique_files']}")
    print(f"File types: {stats['file_types']}")
    
    # Show sample chunks
    if documents:
        print(f"\n=== Sample Chunk ===")
        sample_chunk = documents[0]
        print(f"Source: {sample_chunk.metadata['source_file']}")
        print(f"Content preview: {sample_chunk.page_content[:200]}...")
    
    return documents

if __name__ == "__main__":
    test_document_loader()
