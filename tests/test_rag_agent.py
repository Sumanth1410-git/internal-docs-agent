"""
Comprehensive tests for RAG Agent with advanced query processing
Tests all query types, fallback mechanisms, and performance metrics
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import os
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Import test utilities
from tests import create_test_documents, cleanup_test_files, TEST_DATA_DIR

# Mock environment variables before importing modules
@pytest.fixture(autouse=True)
def mock_env_vars():
    """Mock environment variables for testing"""
    with patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-key-12345',
        'MAX_CHUNK_SIZE': '500',
        'BATCH_SIZE': '3',
        'MAX_MEMORY_USAGE': '1024'
    }):
        yield

@pytest.fixture
def mock_openai():
    """Mock OpenAI services"""
    with patch('langchain_openai.ChatOpenAI') as mock_llm, \
         patch('langchain_openai.OpenAIEmbeddings') as mock_embeddings:
        
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = "Test response from LLM"
        mock_llm.return_value.invoke.return_value = mock_response
        
        # Mock embeddings
        mock_embeddings.return_value.embed_documents.return_value = [
            [0.1, 0.2, 0.3] * 128  # 384-dimensional embedding
        ]
        mock_embeddings.return_value.embed_query.return_value = [0.1, 0.2, 0.3] * 128
        
        yield mock_llm, mock_embeddings

@pytest.fixture
def sample_documents():
    """Create sample documents for testing"""
    create_test_documents()
    yield TEST_DATA_DIR
    cleanup_test_files()

class TestProductionRAGAgent:
    """Test suite for Production RAG Agent"""
    
    def test_rag_agent_initialization(self, mock_openai):
        """Test RAG agent initialization with all components"""
        from rag_agent import ProductionRAGAgent
        
        agent = ProductionRAGAgent(
            openai_api_key="test-key",
            model_name="gpt-3.5-turbo",
            embedding_strategy="openai"
        )
        
        assert agent is not None
        assert agent.model_name == "gpt-3.5-turbo"
        assert agent.openai_api_key == "test-key"
        assert agent.query_stats['total_queries'] == 0
    
    def test_component_initialization(self, mock_openai):
        """Test individual component initialization"""
        from rag_agent import ProductionRAGAgent
        
        agent = ProductionRAGAgent(openai_api_key="test-key")
        
        with patch.object(agent.vector_manager, 'initialize_embedding_service', return_value=True), \
             patch.object(agent.vector_manager, 'initialize_vector_store', return_value=True):
            
            status = agent.initialize_components()
            
            assert status['embeddings'] is True
            assert status['vector_store'] is True
            assert status['llm'] is True
    
    def test_query_with_context_standard(self, mock_openai, sample_documents):
        """Test standard query processing"""
        from rag_agent import ProductionRAGAgent
        from document_loader import OptimizedDocumentLoader
        
        # Load test documents
        loader = OptimizedDocumentLoader()
        documents = loader.load_directory(str(sample_documents))
        
        agent = ProductionRAGAgent(openai_api_key="test-key")
        
        # Mock vector store operations
        with patch.object(agent.vector_manager, 'initialize_embedding_service', return_value=True), \
             patch.object(agent.vector_manager, 'initialize_vector_store', return_value=True), \
             patch.object(agent, '_setup_advanced_retrieval_chain', return_value=True):
            
            # Mock retrieval chain
            mock_result = {
                'result': 'Test policy response',
                'source_documents': documents[:2] if documents else []
            }
            agent.retrieval_chain = Mock()
            agent.retrieval_chain.invoke.return_value = mock_result
            
            # Test query
            result = agent.query_with_context("What's our remote work policy?")
            
            assert 'answer' in result
            assert 'source_documents' in result
            assert 'response_time' in result
            assert 'confidence_score' in result
            assert result['processing_mode'] == 'standard'
    
    def test_advanced_query_processing(self, mock_openai):
        """Test advanced query types (comparison, temporal, conditional)"""
        from rag_agent import ProductionRAGAgent, AdvancedRAGProcessor
        
        agent = ProductionRAGAgent(openai_api_key="test-key")
        
        # Mock advanced processor
        mock_processor = Mock(spec=AdvancedRAGProcessor)
        mock_processor.process_advanced_query.return_value = {
            'answer': 'Advanced comparison analysis result',
            'source_documents': [],
            'query_type': 'comparison',
            'response_time': 1.5,
            'confidence_score': 0.8,
            'comparison_subjects': ['policy A', 'policy B']
        }
        agent.advanced_processor = mock_processor
        
        # Test comparison query
        result = agent.query_with_context(
            "Compare our remote work policy with office policy",
            query_type="comparison"
        )
        
        assert result['processing_mode'] == 'advanced'
        assert 'comparison_subjects' in result
        assert result['query_type'] == 'comparison'
    
    def test_fallback_query_processing(self, mock_openai):
        """Test fallback query when retrieval chain is unavailable"""
        from rag_agent import ProductionRAGAgent
        
        agent = ProductionRAGAgent(openai_api_key="test-key")
        agent.retrieval_chain = None  # Simulate missing retrieval chain
        
        # Mock vector store similarity search
        mock_documents = [Mock(), Mock()]
        mock_documents[0].page_content = "Sample policy content"
        mock_documents[1].page_content = "Additional policy information"
        
        with patch.object(agent.vector_manager, 'similarity_search', 
                         return_value=[(doc, 0.8) for doc in mock_documents]):
            
            result = agent.query_with_context("Test fallback query")
            
            assert result['processing_mode'] == 'fallback'
            assert 'relevant information' in result['answer'].lower()
            assert len(result['source_documents']) == 2
    
    def test_conversation_history_management(self, mock_openai):
        """Test conversation history tracking and context"""
        from rag_agent import ProductionRAGAgent
        
        agent = ProductionRAGAgent(openai_api_key="test-key")
        
        # Mock retrieval chain for multiple queries
        mock_result = {
            'result': 'Response to query',
            'source_documents': []
        }
        agent.retrieval_chain = Mock()
        agent.retrieval_chain.invoke.return_value = mock_result
        
        # Simulate multiple queries to build history
        for i in range(5):
            agent.query_with_context(f"Test query {i}", include_conversation_history=True)
        
        # Check conversation history management
        assert len(agent.conversation_history) <= 10  # Should cap at 10
        assert agent.conversation_history[-1]['question'] == "Test query 4"
    
    def test_document_ranking_and_confidence(self, mock_openai):
        """Test document ranking and confidence scoring"""
        from rag_agent import ProductionRAGAgent
        from langchain.schema import Document
        
        agent = ProductionRAGAgent(openai_api_key="test-key")
        
        # Create test documents
        test_docs = [
            Document(page_content="remote work policy details", metadata={'source_file': 'policy.txt'}),
            Document(page_content="office work requirements", metadata={'source_file': 'handbook.txt'}),
            Document(page_content="general company information", metadata={'source_file': 'info.txt'})
        ]
        
        # Test document ranking
        ranked_docs = agent._rank_source_documents(test_docs, "remote work policy")
        
        assert len(ranked_docs) == 3
        assert isinstance(ranked_docs, list)
        # First document should be most relevant to "remote work policy"
        assert "remote work" in ranked_docs[0].page_content.lower()
    
    def test_performance_metrics_tracking(self, mock_openai):
        """Test query statistics and performance tracking"""
        from rag_agent import ProductionRAGAgent
        
        agent = ProductionRAGAgent(openai_api_key="test-key")
        
        # Mock successful query processing
        with patch.object(agent, '_record_successful_query') as mock_record:
            mock_result = {'result': 'Test response', 'source_documents': []}
            agent.retrieval_chain = Mock()
            agent.retrieval_chain.invoke.return_value = mock_result
            
            agent.query_with_context("Test performance tracking")
            
            # Verify metrics are updated
            assert agent.query_stats['total_queries'] == 1
            mock_record.assert_called_once()
    
    def test_error_handling_and_recovery(self, mock_openai):
        """Test error handling in query processing"""
        from rag_agent import ProductionRAGAgent
        
        agent = ProductionRAGAgent(openai_api_key="test-key")
        
        # Mock retrieval chain to raise an exception
        agent.retrieval_chain = Mock()
        agent.retrieval_chain.invoke.side_effect = Exception("Test exception")
        
        result = agent.query_with_context("Query that causes error")
        
        assert result['processing_mode'] == 'error'
        assert 'error processing your question' in result['answer'].lower()
        assert agent.query_stats['failed_queries'] == 1

class TestAdvancedQueryProcessor:
    """Test suite for Advanced Query Processor"""
    
    def test_query_intent_analysis(self):
        """Test query intent detection and classification"""
        from rag_agent import AdvancedQueryProcessor, QueryType
        
        processor = AdvancedQueryProcessor()
        
        # Test comparison query
        intent = processor.analyze_query("Compare sick leave with vacation policy")
        assert intent.primary_type == QueryType.COMPARISON
        assert len(intent.comparison_subjects) > 0
        
        # Test temporal query
        intent = processor.analyze_query("What changed in our benefits recently?")
        assert intent.primary_type == QueryType.TEMPORAL
        assert len(intent.temporal_indicators) > 0
        
        # Test conditional query
        intent = processor.analyze_query("If I work remotely 3 days, what approval do I need?")
        assert intent.primary_type == QueryType.CONDITIONAL
        assert len(intent.conditions) > 0
    
    def test_comparison_subject_extraction(self):
        """Test extraction of comparison subjects from queries"""
        from rag_agent import AdvancedQueryProcessor
        
        processor = AdvancedQueryProcessor()
        
        subjects = processor._extract_comparison_subjects(
            "Compare remote work policy with office work policy"
        )
        
        assert len(subjects) >= 2
        assert any("remote work" in subject for subject in subjects)
        assert any("office work" in subject for subject in subjects)
    
    def test_temporal_indicator_extraction(self):
        """Test extraction of temporal indicators"""
        from rag_agent import AdvancedQueryProcessor
        
        processor = AdvancedQueryProcessor()
        
        indicators = processor._extract_temporal_indicators(
            "What changed recently in our updated benefits package?"
        )
        
        assert "changed" in indicators or "recently" in indicators or "updated" in indicators
    
    def test_conditional_extraction(self):
        """Test extraction of conditions from conditional queries"""
        from rag_agent import AdvancedQueryProcessor
        
        processor = AdvancedQueryProcessor()
        
        conditions = processor._extract_conditions(
            "If I submit my timesheet late, what happens?"
        )
        
        assert len(conditions) > 0
        assert any("submit" in condition for condition in conditions)

class TestAdvancedRAGProcessor:
    """Test suite for Advanced RAG Processor with complex query handling"""
    
    @pytest.fixture
    def mock_rag_agent(self):
        """Create mock RAG agent for testing"""
        mock_agent = Mock()
        mock_agent.vector_manager.vector_store = Mock()
        mock_agent.vector_manager.similarity_search.return_value = [
            (Mock(page_content="Test content", metadata={'source_file': 'test.txt'}), 0.8)
        ]
        mock_agent.llm = Mock()
        mock_agent.llm.invoke.return_value = Mock(content="Test LLM response")
        return mock_agent
    
    def test_comparison_query_handling(self, mock_rag_agent):
        """Test advanced comparison query processing"""
        from rag_agent import AdvancedRAGProcessor, QueryIntent, QueryType
        
        processor = AdvancedRAGProcessor(mock_rag_agent)
        
        # Create mock intent
        intent = QueryIntent(QueryType.COMPARISON, 0.8)
        intent.comparison_subjects = ["sick leave", "vacation policy"]
        
        result = processor._handle_comparison_query(
            "Compare sick leave with vacation policy", intent
        )
        
        assert result['query_type'] == 'comparison'
        assert 'comparison_subjects' in result
        assert result['confidence_score'] == 0.8
    
    def test_temporal_query_handling(self, mock_rag_agent):
        """Test temporal analysis query processing"""
        from rag_agent import AdvancedRAGProcessor, QueryIntent, QueryType
        
        processor = AdvancedRAGProcessor(mock_rag_agent)
        
        intent = QueryIntent(QueryType.TEMPORAL, 0.7)
        intent.temporal_indicators = ["changed", "recently"]
        
        result = processor._handle_temporal_query(
            "What changed recently in our policies?", intent
        )
        
        assert result['query_type'] == 'temporal'
        assert 'changes_detected' in result
    
    def test_aggregation_query_handling(self, mock_rag_agent):
        """Test aggregation and counting queries"""
        from rag_agent import AdvancedRAGProcessor, QueryIntent, QueryType
        
        processor = AdvancedRAGProcessor(mock_rag_agent)
        
        intent = QueryIntent(QueryType.AGGREGATION, 0.6)
        
        result = processor._handle_aggregation_query(
            "How many benefits do we offer?", intent
        )
        
        assert result['query_type'] == 'aggregation'
        assert 'aggregation analysis' in result['answer'].lower()
        assert 'aggregation_data' in result
    
    def test_statistical_query_handling(self, mock_rag_agent):
        """Test statistical analysis queries"""
        from rag_agent import AdvancedRAGProcessor, QueryIntent, QueryType
        
        processor = AdvancedRAGProcessor(mock_rag_agent)
        
        intent = QueryIntent(QueryType.STATISTICAL, 0.5)
        intent.statistical_operations = ["percentage", "average"]
        
        result = processor._handle_statistical_query(
            "What percentage of employees use remote work?", intent
        )
        
        assert result['query_type'] == 'statistical'
        assert 'statistical_data' in result
    
    def test_document_grouping_for_comparison(self, mock_rag_agent):
        """Test document grouping logic for comparison queries"""
        from rag_agent import AdvancedRAGProcessor
        from langchain.schema import Document
        
        processor = AdvancedRAGProcessor(mock_rag_agent)
        
        # Create test documents
        docs = [
            (Document(page_content="sick leave policy details", metadata={'source_file': 'hr.txt'}), 0.9),
            (Document(page_content="vacation policy information", metadata={'source_file': 'hr.txt'}), 0.8),
            (Document(page_content="general company info", metadata={'source_file': 'general.txt'}), 0.6)
        ]
        
        groups = processor._group_documents_for_comparison(docs, ["sick leave", "vacation"])
        
        assert len(groups) > 0
        # Should group documents by subject relevance
        assert any("sick" in group_name.lower() for group_name in groups.keys()) or \
               any("vacation" in group_name.lower() for group_name in groups.keys())

def test_integration_rag_with_document_loader():
    """Integration test with document loader"""
    from rag_agent import ProductionRAGAgent
    from document_loader import OptimizedDocumentLoader
    
    # Create temporary test documents
    create_test_documents()
    
    try:
        loader = OptimizedDocumentLoader()
        documents = loader.load_directory(str(TEST_DATA_DIR))
        
        # Mock the RAG agent to avoid actual API calls
        with patch('langchain_openai.ChatOpenAI'), \
             patch('langchain_openai.OpenAIEmbeddings'):
            
            agent = ProductionRAGAgent(openai_api_key="test-key")
            
            # Test document addition
            success = agent.add_documents(documents)
            
            # Should not fail even with mocked components
            assert isinstance(success, bool)
            
    finally:
        cleanup_test_files()

@pytest.mark.asyncio
async def test_concurrent_query_processing():
    """Test concurrent query processing capabilities"""
    import asyncio
    from rag_agent import ProductionRAGAgent
    
    with patch('langchain_openai.ChatOpenAI'), \
         patch('langchain_openai.OpenAIEmbeddings'):
        
        agent = ProductionRAGAgent(openai_api_key="test-key")
        
        # Mock retrieval chain
        mock_result = {'result': 'Concurrent test response', 'source_documents': []}
        agent.retrieval_chain = Mock()
        agent.retrieval_chain.invoke.return_value = mock_result
        
        # Create multiple concurrent queries
        async def query_task(query_id):
            return agent.query_with_context(f"Test query {query_id}")
        
        # Execute concurrent queries
        tasks = [query_task(i) for i in range(5)]
        results = await asyncio.gather(*tasks)
        
        # All queries should complete successfully
        assert len(results) == 5
        assert all('answer' in result for result in results)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
