import pytest
import asyncio
import time
from unittest.mock import Mock, MagicMock, patch, call
from datetime import datetime
import logging
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Corrected imports from src directory
from src.slack_bot import EnhancedIntelligentSlackBot, SlackBotMetrics

class TestEnhancedIntelligentSlackBot:
    """Test suite for Enhanced Intelligent Slack Bot"""

    @pytest.fixture
    def mock_slack_app(self):
        """Mock Slack App and WebClient"""
        with patch('src.slack_bot.App') as mock_app, \
             patch('src.slack_bot.WebClient') as mock_client:
            
            mock_app_instance = MagicMock()
            mock_client_instance = MagicMock()
            
            mock_app.return_value = mock_app_instance
            mock_client.return_value = mock_client_instance
            
            # Mock environment variables
            with patch.dict('os.environ', {
                'SLACK_BOT_TOKEN': 'test-bot-token',
                'SLACK_APP_TOKEN': 'test-app-token'
            }):
                yield mock_app_instance, mock_client_instance

    @pytest.fixture
    def mock_rag_agent(self):
        """Mock RAG agent"""
        mock_agent = Mock()
        mock_agent.initialize_components.return_value = {
            'vector_store': True,
            'embeddings': True,
            'llm': True,
            'retrieval_chain': True
        }
        mock_agent.query_with_context.return_value = {
            'answer': 'Test response from RAG agent',
            'source_documents': [],
            'response_time': 1.0,
            'confidence_score': 0.8
        }
        return mock_agent

    def test_bot_initialization(self, mock_slack_app, mock_rag_agent):
        """Test bot initialization with all components"""
        with patch('src.slack_bot.ProductionRAGAgent', return_value=mock_rag_agent), \
             patch('src.slack_bot.SlackFileProcessor'), \
             patch('src.slack_bot.EnterpriseApprovalWorkflow'), \
             patch('src.slack_bot.MultilingualSupport'):
            
            bot = EnhancedIntelligentSlackBot()
            
            # Verify bot initialization
            assert bot.bot_info['name'] == 'Internal Docs AI'
            assert 'emoji' in bot.bot_info
            assert len(bot.bot_info['capabilities']) > 0
            
            # Verify components are initialized
            assert hasattr(bot, 'metrics')
            assert isinstance(bot.metrics, SlackBotMetrics)

    def test_component_initialization_status(self, mock_slack_app):
        """Test component initialization with different availability states"""
        # Test with all components available
        with patch('src.slack_bot.FILE_UPLOAD_AVAILABLE', True), \
             patch('src.slack_bot.APPROVAL_AVAILABLE', True), \
             patch('src.slack_bot.MULTILINGUAL_AVAILABLE', True), \
             patch('src.slack_bot.RAG_AVAILABLE', True):
            
            with patch('src.slack_bot.SlackFileProcessor') as mock_fp, \
                 patch('src.slack_bot.EnterpriseApprovalWorkflow') as mock_aw, \
                 patch('src.slack_bot.MultilingualSupport') as mock_ml, \
                 patch('src.slack_bot.ProductionRAGAgent') as mock_rag:
                
                bot = EnhancedIntelligentSlackBot()
                
                # Verify all components are attempted to be initialized
                mock_fp.assert_called_once()
                mock_aw.assert_called_once()
                mock_ml.assert_called_once()
                mock_rag.assert_called_once()

    def test_message_processing_with_language_detection(self, mock_slack_app, mock_rag_agent):
        """Test message processing with language detection"""
        with patch('src.slack_bot.ProductionRAGAgent', return_value=mock_rag_agent):
            mock_multilingual = Mock()
            mock_multilingual.detect_language.return_value = ('en', 0.9)
            
            with patch('src.slack_bot.MultilingualSupport', return_value=mock_multilingual):
                bot = EnhancedIntelligentSlackBot()
                
                # Test message processing
                mock_event = {
                    'user': 'U123456',
                    'channel': 'C123456',
                    'text': 'What is our refund policy?'
                }
                
                mock_say = Mock()
                
                # Process message
                bot._process_message_enhanced(mock_event, mock_say, {}, False)
                
                # Verify language detection was called
                mock_multilingual.detect_language.assert_called_once()
                
                # Verify say was called with a response
                mock_say.assert_called_once()

    def test_advanced_query_classification(self, mock_slack_app, mock_rag_agent):
        """Test enhanced query classification for advanced types"""
        with patch('src.slack_bot.ProductionRAGAgent', return_value=mock_rag_agent):
            bot = EnhancedIntelligentSlackBot()
            
            # Test different query types
            test_queries = {
                "What's the difference between sick leave and PTO?": 'comparison',
                "What changed in our benefits recently?": 'temporal',
                "If I work remotely 3 days, what's required?": 'conditional',
                "Analyze our remote work policies": 'analytical'
            }
            
            for query, expected_type in test_queries.items():
                result = bot._classify_query_enhanced(query)
                assert result == expected_type, f"Query '{query}' should be classified as '{expected_type}', got '{result}'"

    def test_file_upload_processing(self, mock_slack_app):
        """Test file upload event handling"""
        # Mock file processor
        mock_file_processor = Mock()
        mock_file_processor.process_uploaded_file.return_value = (
            True, "File processed successfully", []
        )
        
        with patch('src.slack_bot.SlackFileProcessor', return_value=mock_file_processor):
            bot = EnhancedIntelligentSlackBot()
            
            # Verify file processor is initialized
            assert bot.file_processor is not None
            assert bot.file_processor == mock_file_processor

    def test_approval_workflow_integration(self, mock_slack_app):
        """Test approval workflow integration"""
        # Mock approval system
        mock_approval_system = Mock()
        mock_approval_system.create_approval_request.return_value = "req_12345"
        mock_approval_system.process_approval_response.return_value = {
            'success': True,
            'status': 'approved'
        }
        
        with patch('src.slack_bot.EnterpriseApprovalWorkflow', return_value=mock_approval_system):
            bot = EnhancedIntelligentSlackBot()
            
            # Test approval request creation
            mock_body = {
                'user': {'id': 'U123456'},
                'actions': [{'value': 'approve_req_12345'}]
            }
            mock_say = Mock()
            
            bot._handle_approval_action(mock_body, mock_say, 'approve')
            
            # Verify approval processing was called
            mock_approval_system.process_approval_response.assert_called_once()

    def test_multilingual_response_formatting(self, mock_slack_app):
        """Test multilingual response formatting"""
        # Mock multilingual support
        mock_multilingual = Mock()
        mock_multilingual.format_response_with_language.return_value = "Respuesta en español"
        
        with patch('src.slack_bot.MultilingualSupport', return_value=mock_multilingual):
            bot = EnhancedIntelligentSlackBot()
            
            # Test response formatting
            test_result = {
                'answer': 'Test answer',
                'source_documents': [],
                'response_time': 1.0,
                'confidence_score': 0.8
            }
            
            response = bot._format_response(test_result, {}, 'general', 'es')
            
            # Should use multilingual formatting for non-English
            mock_multilingual.format_response_with_language.assert_called_once()

    def test_rate_limiting(self, mock_slack_app):
        """Test rate limiting functionality"""
        bot = EnhancedIntelligentSlackBot()
        
        user_id = "U123456"
        
        # First 10 requests should be allowed
        for i in range(10):
            assert not bot._is_rate_limited(user_id)
        
        # 11th request should be rate limited
        assert bot._is_rate_limited(user_id)

    def test_error_handling_in_message_processing(self, mock_slack_app):
        """Test error handling during message processing"""
        bot = EnhancedIntelligentSlackBot()
        
        # Test with invalid event data
        invalid_event = {}
        mock_say = Mock()
        
        # This should not raise an exception
        bot._process_message_enhanced(invalid_event, mock_say, {}, False)
        
        # Verify error response was sent
        mock_say.assert_called_once()
        call_args = mock_say.call_args[0][0]
        assert "error" in call_args.lower() or "technical difficulties" in call_args.lower()

    def test_advanced_response_formatting(self, mock_slack_app):
        """Test advanced response formatting for different query types"""
        bot = EnhancedIntelligentSlackBot()
        
        # Mock advanced processor result
        advanced_result = {
            'answer': 'Detailed comparison analysis',
            'source_documents': [Mock(page_content="Test content", metadata={'source_file': 'test.txt'})],
            'query_type': 'comparison',
            'response_time': 2.0,
            'confidence_score': 0.9
        }
        
        # Test advanced response formatting
        response = bot._format_advanced_response(advanced_result, 'comparison')
        
        assert 'Comparison Analysis' in response
        assert 'Detailed comparison analysis' in response
        assert 'Sources:' in response

    def test_user_context_management(self, mock_slack_app):
        """Test user context tracking and management"""
        bot = EnhancedIntelligentSlackBot()
        
        user_id = "U123456"
        
        # Test context creation
        context = bot._get_user_context(user_id)
        assert 'first_interaction' in context
        assert context['message_count'] == 0
        
        # Test context updating
        bot._update_user_context(user_id, "Test question", "Test answer")
        updated_context = bot._get_user_context(user_id)
        assert updated_context['message_count'] == 1
        assert updated_context['last_question'] == "Test question"

    def test_metrics_tracking(self, mock_slack_app):
        """Test metrics tracking functionality"""
        metrics = SlackBotMetrics()
        
        # Test interaction recording
        metrics.record_interaction('U123', 'C123', 'general', 1.5, True)
        
        summary = metrics.get_summary()
        
        assert summary['total_messages'] == 1
        assert summary['successful_responses'] == 1
        assert summary['average_response_time'] == 1.5
        assert 'U123' in summary['top_users']

    def test_slash_command_handling(self, mock_slack_app, mock_rag_agent):
        """Test slash command processing"""
        with patch('src.slack_bot.ProductionRAGAgent', return_value=mock_rag_agent):
            bot = EnhancedIntelligentSlackBot()
            
            mock_command = {
                'text': 'What are our employee benefits?',
                'user_id': 'U123456'
            }
            mock_respond = Mock()
            
            bot._handle_slash_command(mock_respond, mock_command)
            
            # Verify response was sent
            mock_respond.assert_called_once()

    def test_help_message_generation(self, mock_slack_app):
        """Test help message generation with all features"""
        bot = EnhancedIntelligentSlackBot()
        
        mock_respond = Mock()
        bot._send_help_message(mock_respond)
        
        # Verify help message was sent
        mock_respond.assert_called_once()
        
        help_content = mock_respond.call_args[0][0]
        assert 'Internal Docs AI' in help_content
        assert 'capabilities' in help_content.lower() or 'help' in help_content.lower()


class TestSlackBotIntegration:
    """Integration tests for Slack bot components"""

    @pytest.fixture
    def mock_slack_app(self):
        """Mock Slack App and WebClient for integration tests"""
        with patch('src.slack_bot.App') as mock_app, \
             patch('src.slack_bot.WebClient') as mock_client:
            
            mock_app_instance = MagicMock()
            mock_client_instance = MagicMock()
            
            mock_app.return_value = mock_app_instance
            mock_client.return_value = mock_client_instance
            
            with patch.dict('os.environ', {
                'SLACK_BOT_TOKEN': 'test-bot-token',
                'SLACK_APP_TOKEN': 'test-app-token'
            }):
                yield mock_app_instance, mock_client_instance

    def test_integration_with_rag_agent(self, mock_slack_app):
        """Test integration between Slack bot and RAG agent"""
        # Mock RAG agent with realistic responses
        mock_rag_agent = Mock()
        mock_rag_agent.query_with_context.return_value = {
            'answer': 'According to our company policy, remote work is allowed up to 3 days per week.',
            'source_documents': [Mock(page_content="Remote work policy", metadata={'source_file': 'policy.txt'})],
            'response_time': 1.2,
            'confidence_score': 0.85,
            'mode': 'production'
        }

        with patch('src.slack_bot.ProductionRAGAgent', return_value=mock_rag_agent):
            bot = EnhancedIntelligentSlackBot()
            
            # Test query processing through bot
            mock_event = {
                'user': 'U123456',
                'channel': 'C123456',
                'text': 'What is our remote work policy?'
            }
            
            mock_say = Mock()
            bot._process_message_enhanced(mock_event, mock_say, {}, False)
            
            # Verify RAG agent was called
            mock_rag_agent.query_with_context.assert_called()
            
            # Verify response was sent
            mock_say.assert_called_once()

    def test_bot_startup_and_connection(self, mock_slack_app):
        """Test bot startup and Slack connection with proper mocking"""
        with patch('slack_bolt.adapter.socket_mode.SocketModeHandler') as mock_handler:
            # Mock the auth_test method
            mock_app_instance, mock_client_instance = mock_slack_app
            mock_client_instance.auth_test.return_value = {
                'user': 'TestBot', 
                'user_id': 'U123456789'
            }
            
            # Create bot instance
            bot = EnhancedIntelligentSlackBot()
            
            # Configure handler mock
            mock_handler_instance = MagicMock()
            mock_handler.return_value = mock_handler_instance
            
            try:
                # Test startup - this may fail at handler.start(), which is expected
                bot.start()
            except Exception as e:
                # Expected to fail at handler.start() in test environment
                pass
            
            # Verify auth_test was called
            mock_client_instance.auth_test.assert_called_once()
            
            # Verify handler was created
            mock_handler.assert_called_once()


@pytest.mark.asyncio
async def test_concurrent_message_handling():
    """Test handling multiple concurrent messages"""
    with patch('src.slack_bot.App'), \
         patch('src.slack_bot.WebClient'), \
         patch('src.slack_bot.ProductionRAGAgent') as mock_rag:
        
        with patch.dict('os.environ', {
            'SLACK_BOT_TOKEN': 'test-bot-token',
            'SLACK_APP_TOKEN': 'test-app-token'
        }):
            
            # Mock RAG agent
            mock_rag_instance = Mock()
            mock_rag_instance.query_with_context.return_value = {
                'answer': 'Test response',
                'source_documents': [],
                'response_time': 0.5,
                'confidence_score': 0.8
            }
            mock_rag.return_value = mock_rag_instance
            
            bot = EnhancedIntelligentSlackBot()
            
            # Simulate multiple concurrent messages
            events = [
                {'user': f'U{i}', 'channel': 'C123', 'text': f'Question {i}'}
                for i in range(5)
            ]
            
            mock_says = [Mock() for _ in range(5)]
            
            # Process messages concurrently
            tasks = []
            for i, (event, say) in enumerate(zip(events, mock_says)):
                # Use asyncio.create_task for true concurrency
                task = asyncio.create_task(
                    asyncio.to_thread(bot._process_message_enhanced, event, say, {}, False)
                )
                tasks.append(task)
            
            # Wait for all tasks to complete
            await asyncio.gather(*tasks)
            
            # Verify all messages were processed
            for say in mock_says:
                say.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
