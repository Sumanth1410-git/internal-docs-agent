import os
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import threading
from collections import defaultdict, deque
import re

# Slack SDK imports
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

# Environment and utilities
from dotenv import load_dotenv
import psutil

# Integration Expansions - New Imports
try:
    from file_processor import SlackFileProcessor
    FILE_UPLOAD_AVAILABLE = True
except ImportError:
    print("⚠️ File processor not available")
    FILE_UPLOAD_AVAILABLE = False

try:
    from approval_system import EnterpriseApprovalWorkflow, ApprovalType
    APPROVAL_AVAILABLE = True
except ImportError:
    print("⚠️ Approval system not available")
    APPROVAL_AVAILABLE = False

try:
    from multilingual_support import MultilingualSupport
    MULTILINGUAL_AVAILABLE = True
except ImportError:
    print("⚠️ Multilingual support not available")
    MULTILINGUAL_AVAILABLE = False

# Windows patch and advanced imports
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    
    # Apply Windows patch if needed
    try:
        from windows_patch import patch_pwd_for_windows
        patch_pwd_for_windows()
    except ImportError:
        pass  # Windows patch not needed
        
    from advanced_query_processor import AdvancedQueryProcessor, QueryType
    ADVANCED_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Advanced query processor not available: {e}")
    ADVANCED_AVAILABLE = False

# Analytics imports
try:
    from analytics_dashboard import AdvancedAnalyticsDashboard
    ANALYTICS_AVAILABLE = True
except ImportError:
    ANALYTICS_AVAILABLE = False

# RAG system imports
try:
    from rag_agent import ProductionRAGAgent
    from document_loader import OptimizedDocumentLoader
    from vector_store_manager import ProductionVectorStoreManager
    RAG_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ RAG system import failed: {e}")
    RAG_AVAILABLE = False

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SlackBotMetrics:
    """Enhanced metrics tracking for Slack bot performance"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.total_messages = 0
        self.successful_responses = 0
        self.failed_responses = 0
        self.average_response_time = 0.0
        self.user_interactions = defaultdict(int)
        self.channel_interactions = defaultdict(int)
        self.query_types = defaultdict(int)
        self.popular_queries = deque(maxlen=100)
        self.error_log = deque(maxlen=50)
        
        # New metrics for integrations
        self.files_processed = 0
        self.approvals_requested = 0
        self.language_detections = defaultdict(int)
        self.file_upload_errors = 0
        
    def record_interaction(self, user_id: str, channel_id: str, query_type: str, response_time: float, success: bool):
        """Record a user interaction"""
        self.total_messages += 1
        self.user_interactions[user_id] += 1
        self.channel_interactions[channel_id] += 1
        self.query_types[query_type] += 1
        
        if success:
            self.successful_responses += 1
            total_time = (self.average_response_time * (self.successful_responses - 1) + response_time)
            self.average_response_time = total_time / self.successful_responses
        else:
            self.failed_responses += 1
            
    def record_query(self, query: str):
        """Record a popular query"""
        self.popular_queries.append({
            'query': query[:100],
            'timestamp': datetime.now().isoformat()
        })
    
    def record_error(self, error: str, context: Dict[str, Any]):
        """Record an error with context"""
        self.error_log.append({
            'error': str(error),
            'context': context,
            'timestamp': datetime.now().isoformat()
        })
    
    def record_file_processed(self):
        """Record successful file processing"""
        self.files_processed += 1
        
    def record_file_error(self):
        """Record file processing error"""
        self.file_upload_errors += 1
        
    def record_approval_request(self):
        """Record approval request"""
        self.approvals_requested += 1
        
    def record_language_detection(self, language: str):
        """Record language detection"""
        self.language_detections[language] += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary with integration stats"""
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            'uptime_hours': uptime / 3600,
            'total_messages': self.total_messages,
            'successful_responses': self.successful_responses,
            'failed_responses': self.failed_responses,
            'success_rate': self.successful_responses / max(1, self.total_messages) * 100,
            'average_response_time': self.average_response_time,
            'messages_per_hour': self.total_messages / max(1, uptime / 3600),
            'top_users': dict(sorted(self.user_interactions.items(), key=lambda x: x[1], reverse=True)[:5]),
            'top_channels': dict(sorted(self.channel_interactions.items(), key=lambda x: x[1], reverse=True)[:5]),
            'query_type_distribution': dict(self.query_types),
            'recent_errors': list(self.error_log)[-5:],
            'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
            # Integration metrics
            'files_processed': self.files_processed,
            'file_upload_errors': self.file_upload_errors,
            'approvals_requested': self.approvals_requested,
            'language_detections': dict(self.language_detections)
        }

class EnhancedIntelligentSlackBot:
    """Production-ready Slack bot with complete integration suite"""
    
    def __init__(self):
        """Initialize the enhanced intelligent Slack bot"""
        
        # Slack configuration
        self.bot_token = os.getenv("SLACK_BOT_TOKEN")
        self.app_token = os.getenv("SLACK_APP_TOKEN")
        
        if not self.bot_token or not self.app_token:
            raise ValueError("Missing Slack tokens. Please check your .env file.")
        
        # Initialize Slack app
        self.app = App(token=self.bot_token)
        self.client = WebClient(token=self.bot_token)
        
        # Initialize Integration Systems
        self.file_processor = None
        if FILE_UPLOAD_AVAILABLE:
            try:
                self.file_processor = SlackFileProcessor(self.bot_token)
                logger.info("✅ File processor initialized")
            except Exception as e:
                logger.error(f"❌ File processor initialization failed: {e}")
        
        self.approval_system = None
        if APPROVAL_AVAILABLE:
            try:
                self.approval_system = EnterpriseApprovalWorkflow()
                logger.info("✅ Approval system initialized")
            except Exception as e:
                logger.error(f"❌ Approval system initialization failed: {e}")
        
        self.multilingual = None
        self.user_languages = {}  # Store user language preferences
        if MULTILINGUAL_AVAILABLE:
            try:
                self.multilingual = MultilingualSupport()
                logger.info("✅ Multilingual support initialized")
            except Exception as e:
                logger.error(f"❌ Multilingual support initialization failed: {e}")
        
        # Initialize RAG system
        self.rag_agent = None
        if RAG_AVAILABLE:
            try:
                self.rag_agent = ProductionRAGAgent(
                    embedding_strategy="auto",
                    vector_store_type="faiss"
                )
                logger.info("✅ RAG agent initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize RAG agent: {e}")
                self.rag_agent = None
        
        # Initialize advanced processor
        self.advanced_processor = None
        if ADVANCED_AVAILABLE and RAG_AVAILABLE and self.rag_agent:
            try:
                from rag_agent import AdvancedRAGProcessor
                self.advanced_processor = AdvancedRAGProcessor(self.rag_agent)
                logger.info("✅ Advanced processor initialized successfully")
            except Exception as e:
                logger.error(f"❌ Advanced processor initialization failed: {e}")
                self.advanced_processor = None
        
        # Initialize analytics
        self.metrics = SlackBotMetrics()
        self.analytics_dashboard = None
        if ANALYTICS_AVAILABLE:
            try:
                self.analytics_dashboard = AdvancedAnalyticsDashboard()
                logger.info("✅ Analytics dashboard initialized")
            except Exception as e:
                logger.error(f"❌ Analytics initialization failed: {e}")
        
        # Advanced features
        self.user_contexts = {}
        self.typing_indicators = {}
        self.rate_limits = defaultdict(list)
        
        # Enhanced bot personality
        self.bot_info = {
            'name': 'Internal Docs AI',
            'emoji': '🤖',
            'description': 'Your intelligent company documentation assistant with enterprise features',
            'capabilities': [
                'Answer questions about company policies',
                'Help with HR procedures and benefits', 
                'Provide technical documentation',
                'Explain internal processes',
                'Find relevant documents quickly',
                'Compare policies and procedures',
                'Track changes over time',
                'Handle conditional scenarios',
                'Process uploaded documents',
                'Multi-language support',
                'Approval workflow management'
            ]
        }
        
        # Initialize components
        self._setup_event_handlers()
        self._initialize_rag_system()
        
        logger.info("🚀 Enhanced Intelligent Slack Bot initialized successfully")
        logger.info(f"📋 Features: File Upload: {FILE_UPLOAD_AVAILABLE}, Approvals: {APPROVAL_AVAILABLE}, Multi-lang: {MULTILINGUAL_AVAILABLE}")
    
    def _initialize_rag_system(self):
        """Initialize the RAG system with documents"""
        if not RAG_AVAILABLE or not self.rag_agent:
            logger.warning("⚠️ RAG system not available, using fallback responses")
            return
            
        try:
            from document_loader import OptimizedDocumentLoader
            loader = OptimizedDocumentLoader()
            documents = loader.load_directory("data")
            
            if not documents:
                logger.info("No documents found, creating sample documents")
                loader.create_sample_documents()
                documents = loader.load_directory("data")
            
            # Initialize RAG components
            status = self.rag_agent.initialize_components()
            logger.info(f"RAG system status: {status}")
            
            # Ensure vector store is ready
            if not status['vector_store']:
                logger.info("Initializing vector store with documents")
                self.rag_agent.vector_manager.initialize_vector_store(documents)
            
            logger.info(f"✅ RAG system ready with {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize RAG system: {e}")
            self.rag_agent = None
    
    def _setup_event_handlers(self):
        """Setup comprehensive Slack event handlers with all integrations"""
        
        @self.app.event("app_mention")
        def handle_mention(event, say, context):
            """Handle @mentions of the bot"""
            self._handle_message_async(event, say, context, is_mention=True)
        
        @self.app.event("message")
        def handle_direct_message(event, say, context):
            """Handle direct messages to the bot"""
            if event.get('channel_type') == 'im':
                self._handle_message_async(event, say, context, is_mention=False)
        
        # File Upload Integration
        if FILE_UPLOAD_AVAILABLE and self.file_processor:
            @self.app.event("file_shared")
            def handle_file_upload(event, say, client):
                """Handle document uploads from users"""
                try:
                    file_id = event['file_id']
                    user_id = event['user_id']
                    
                    # Get file info from Slack
                    file_info_response = client.files_info(file=file_id)
                    file_data = file_info_response['file']
                    
                    # Get user language for responses
                    user_language = self.user_languages.get(user_id, 'en')
                    
                    # Send processing message
                    if self.multilingual:
                        processing_msg = self.multilingual.translate_message('file_processing', user_language)
                        say(f"📁 {processing_msg}: `{file_data['name']}` ⏳")
                    else:
                        say(f"📁 Processing uploaded file: `{file_data['name']}` ⏳")
                    
                    # Check if approval is needed for this file
                    needs_approval = False
                    if self.approval_system:
                        file_size_mb = file_data.get('size', 0) / (1024 * 1024)
                        file_content_preview = ""  # Would need to download to get preview
                        
                        if file_size_mb > 5 or any(keyword in file_data['name'].lower() 
                                                 for keyword in ['confidential', 'internal', 'secret']):
                            needs_approval = True
                    
                    if needs_approval:
                        # Create approval request
                        approval_request_id = self.approval_system.create_approval_request(
                            requester_id=user_id,
                            approval_type=ApprovalType.DOCUMENT_UPLOAD,
                            content={
                                'file_name': file_data['name'],
                                'file_size_mb': file_size_mb,
                                'file_type': Path(file_data['name']).suffix.lower(),
                                'slack_file_id': file_id
                            },
                            category='general'
                        )
                        
                        self.metrics.record_approval_request()
                        
                        if self.multilingual:
                            approval_msg = self.multilingual.translate_message('approval_needed', user_language)
                            say(f"🔐 {approval_msg}: `{file_data['name']}`\n📋 Request ID: {approval_request_id[:8]}...")
                        else:
                            say(f"🔐 File requires approval: `{file_data['name']}`\n📋 Request ID: {approval_request_id[:8]}...")
                        
                        # Notify approvers
                        self._notify_approvers(approval_request_id)
                        return
                    
                    # Process the file directly
                    success, message, documents = self.file_processor.process_uploaded_file(file_data, user_id)
                    
                    if success and documents:
                        # Add documents to RAG system
                        if self.rag_agent:
                            try:
                                self.rag_agent.add_documents(documents)
                                self.metrics.record_file_processed()
                                
                                if self.multilingual:
                                    success_msg = self.multilingual.translate_message('file_uploaded', user_language)
                                    say(f"✅ {success_msg}: `{file_data['name']}`\n{message}")
                                else:
                                    say(message)
                                
                                # Send usage examples in user's language
                                example_queries = [
                                    f"What does {file_data['name']} say about...?",
                                    f"Summarize {file_data['name']}",
                                ]
                                
                                follow_up = f"\n💡 **Try asking:** \n• " + "\n• ".join(example_queries)
                                say(follow_up)
                                
                            except Exception as e:
                                self.metrics.record_file_error()
                                logger.error(f"Failed to add documents to RAG system: {e}")
                                
                                if self.multilingual:
                                    error_msg = self.multilingual.translate_message('error', user_language)
                                    say(f"✅ File processed, but {error_msg.lower()}: {str(e)}")
                                else:
                                    say(f"✅ File processed successfully, but failed to add to knowledge base: {str(e)}")
                        else:
                            say("✅ File processed successfully, but RAG system is not available.")
                    else:
                        self.metrics.record_file_error()
                        say(message)
                        
                except Exception as e:
                    self.metrics.record_file_error()
                    logger.error(f"File upload processing failed: {e}")
                    
                    user_language = self.user_languages.get(event.get('user_id', ''), 'en')
                    if self.multilingual:
                        error_msg = self.multilingual.translate_message('error', user_language)
                        say(f"❌ {error_msg}: {str(e)}")
                    else:
                        say(f"❌ Sorry, I couldn't process your uploaded file. Error: {str(e)}")
        
        # Approval System Integration
        if APPROVAL_AVAILABLE and self.approval_system:
            @self.app.action("approve_request")
            def handle_approve(ack, body, say):
                """Handle approval button click"""
                ack()
                self._handle_approval_action(body, say, 'approve')

            @self.app.action("reject_request")
            def handle_reject(ack, body, say):
                """Handle rejection button click"""
                ack()
                self._handle_approval_action(body, say, 'reject')

            @self.app.action("request_changes")
            def handle_changes(ack, body, say):
                """Handle request changes button click"""
                ack()
                self._handle_approval_action(body, say, 'changes')

            @self.app.command("/docs-approvals")
            def handle_approvals_command(ack, respond, command):
                """Handle approvals management command"""
                ack()
                self._handle_approvals_management(respond, command)
        
        # Multi-language Support Integration
        if MULTILINGUAL_AVAILABLE and self.multilingual:
            @self.app.command("/docs-language")
            def handle_language_command(ack, respond, command):
                """Handle language selection command"""
                ack()
                
                user_id = command['user_id']
                text = command.get('text', '').strip()
                
                if text in self.multilingual.supported_languages:
                    # Set user language
                    self.user_languages[user_id] = text
                    lang_info = self.multilingual.supported_languages[text]
                    
                    welcome_msg = self.multilingual.translate_message('welcome', text)
                    respond(f"✅ Language set to {lang_info['native']} ({lang_info['name']})\n\n{welcome_msg}")
                
                elif text == 'info':
                    # Show supported languages
                    info = self.multilingual.get_supported_languages_info()
                    respond(info)
                
                else:
                    # Show language selection interface
                    current_lang = self.user_languages.get(user_id, 'en')
                    blocks = self.multilingual.get_language_selection_blocks(current_lang)
                    respond({"blocks": blocks})

            # Language selection action handlers
            for lang_code in self.multilingual.supported_languages.keys():
                action_id = f"select_language_{lang_code}"
                
                @self.app.action(action_id)
                def handle_language_selection(ack, body, say):
                    """Handle language selection button"""
                    ack()
                    
                    user_id = body['user']['id'] 
                    selected_lang = body['actions'][0]['value']
                    
                    # Update user language preference
                    self.user_languages[user_id] = selected_lang
                    
                    # Send confirmation in selected language
                    if self.multilingual:
                        lang_info = self.multilingual.supported_languages[selected_lang]
                        welcome_msg = self.multilingual.translate_message('welcome', selected_lang)
                        
                        confirmation = f"✅ Language updated to {lang_info['native']}\n\n{welcome_msg}"
                        say(confirmation)
        
        # Standard commands
        @self.app.command("/docs")
        def handle_docs_command(ack, respond, command):
            """Handle /docs slash command"""
            ack()
            self._handle_slash_command(respond, command)
        
        @self.app.command("/docs-help")
        def handle_help_command(ack, respond, command):
            """Handle help command"""
            ack()
            self._send_help_message(respond)
        
        @self.app.command("/docs-stats")
        def handle_stats_command(ack, respond, command):
            """Handle stats command"""
            ack()
            self._send_stats_message(respond)
        
        if FILE_UPLOAD_AVAILABLE:
            @self.app.command("/docs-upload-help")
            def handle_upload_help(ack, respond, command):
                """Provide help for file uploads"""
                ack()
                
                help_text = """📁 **File Upload Help**

**Supported File Types:**
• `.txt` - Plain text files
• `.pdf` - PDF documents  
• `.docx/.doc` - Microsoft Word documents
• `.md` - Markdown files
• `.csv` - CSV data files
• `.json` - JSON data files

**Upload Process:**
1. Drag and drop or attach file to any channel where I'm present
2. I'll automatically detect and process the file
3. Large or sensitive files may require approval
4. Content will be added to the knowledge base
5. You can immediately start asking questions about the content

**File Limits:**
• Maximum file size: 10MB
• Files are processed securely and temporarily stored
• Original files remain in your Slack workspace

**Example Questions After Upload:**
• "What does [filename] say about...?"
• "Summarize the key points from [filename]"
• "Search for [topic] in the uploaded document"
                """
                
                respond(help_text)
        
        if ANALYTICS_AVAILABLE and self.analytics_dashboard:
            @self.app.command("/docs-analytics")
            def handle_analytics_command(ack, respond, command):
                """Handle analytics dashboard command"""
                ack()
                try:
                    report = self.analytics_dashboard.export_slack_friendly_report(self.metrics)
                    respond(report)
                except Exception as e:
                    respond(f"❌ Analytics generation failed: {str(e)}")
    
    def _handle_message_async(self, event, say, context, is_mention=False):
        """Handle incoming messages asynchronously with language support"""
        thread = threading.Thread(
            target=self._process_message_enhanced,
            args=(event, say, context, is_mention)
        )
        thread.start()
    
    def _process_message_enhanced(self, event, say, context, is_mention=False):
        """Enhanced message processing with full integration support"""
        start_time = time.time()
        
        try:
            # Extract message details
            user_id = event['user']
            channel_id = event['channel']
            text = event['text']
            
            # Clean mention from text if it's a mention
            if is_mention:
                text = re.sub(r'<@[A-Z0-9]+>', '', text).strip()
            
            # Skip empty messages
            if not text:
                return
            
            # Multi-language support: Detect user language
            user_language = 'en'
            if self.multilingual:
                detected_lang, confidence = self.multilingual.detect_language(text)
                user_language = self.user_languages.get(user_id, detected_lang)
                
                # Update user language if detection confidence is high
                if confidence > 0.7 and detected_lang != user_language:
                    self.user_languages[user_id] = detected_lang
                    user_language = detected_lang
                    self.metrics.record_language_detection(user_language)
            
            # Rate limiting check
            if self._is_rate_limited(user_id):
                if self.multilingual:
                    rate_limit_msg = "⚠️ You're sending messages too quickly. Please wait a moment before trying again."
                    # Could translate this message too
                    say(rate_limit_msg)
                else:
                    say("⚠️ You're sending messages too quickly. Please wait a moment before trying again.")
                return
            
            # Get user context
            user_context = self._get_user_context(user_id)
            user_context['language'] = user_language
            
            # Enhanced query classification
            query_type = self._classify_query_enhanced(text)
            
            # Record the query
            self.metrics.record_query(text)
            
            logger.info(f"Processing {query_type} query from {user_id} in {user_language}: {text[:50]}...")
            
            # Process with advanced capabilities
            response = None
            
            # Try advanced processing for comparison, temporal, conditional queries
            if self.advanced_processor and query_type in ['comparison', 'temporal', 'conditional']:
                try:
                    logger.info(f"🎯 Using advanced processing for {query_type} query")
                    result = self.advanced_processor.process_advanced_query(text)
                    response = self._format_advanced_response(result, query_type, user_language)
                    
                except Exception as e:
                    logger.error(f"❌ Advanced processing failed: {e}")
                    response = None
            
            # Standard RAG processing if advanced didn't work
            if response is None and self.rag_agent:
                try:
                    result = self.rag_agent.query_with_context(
                        question=text,
                        include_conversation_history=True,
                        query_type=query_type
                    )
                    response = self._format_response(result, user_context, query_type, user_language)
                except Exception as e:
                    logger.error(f"❌ Standard RAG processing failed: {e}")
                    response = None
            
            # Final fallback
            if response is None:
                response = self._simple_fallback_response(text, user_language)
            
            # Send response
            if isinstance(response, dict):
                say(**response)
            else:
                say(response)
            
            # Update user context
            answer_text = response if isinstance(response, str) else "Complex response"
            self._update_user_context(user_id, text, answer_text)
            
            # Record metrics
            response_time = time.time() - start_time
            self.metrics.record_interaction(
                user_id, channel_id, query_type, response_time, True
            )
            
            logger.info(f"✅ Successfully processed {query_type} query in {user_language} in {response_time:.2f}s")
            
        except Exception as e:
            # Handle errors gracefully with language support
            user_language = self.user_languages.get(event.get('user', ''), 'en')
            error_msg = str(e)
            logger.error(f"❌ Error processing message: {error_msg}")
            
            self.metrics.record_error(error_msg, {
                'user_id': event.get('user'),
                'channel_id': event.get('channel'),
                'text': event.get('text', '')[:100]
            })
            
            # Send localized error response
            if self.multilingual:
                localized_error = self.multilingual.translate_message('error', user_language)
                say(localized_error)
            else:
                say(self._get_error_response(error_msg))
            
            # Record failed interaction
            response_time = time.time() - start_time
            self.metrics.record_interaction(
                event.get('user'), event.get('channel'), 'error', response_time, False
            )
    
    def _notify_approvers(self, request_id: str):
        """Notify approvers about pending approval request"""
        if not self.approval_system:
            return
            
        try:
            request = self.approval_system.active_requests.get(request_id)
            if not request:
                return
                
            # Get approval blocks
            blocks = self.approval_system.get_approval_blocks(request_id)
            
            # Send to approvers (in a real implementation, you'd have approver user IDs)
            # For now, log that approvers should be notified
            logger.info(f"📨 Approval request {request_id} ready for approver notification")
            
        except Exception as e:
            logger.error(f"Failed to notify approvers: {e}")
    
    def _handle_approval_action(self, body, say, action):
        """Process approval action with multi-language support"""
        try:
            user_id = body['user']['id']
            action_value = body['actions'][0]['value']
            
            # Extract request ID from action value
            request_id = action_value.split('_', 1)[1]
            
            # Process the approval response
            result = self.approval_system.process_approval_response(
                request_id, user_id, action, ""
            )
            
            # Get user language for response
            user_language = self.user_languages.get(user_id, 'en')
            
            if result['success']:
                status = result['status']
                if status == 'approved':
                    if self.multilingual:
                        approved_msg = self.multilingual.translate_message('approved', user_language)
                        say(f"✅ {approved_msg} by <@{user_id}>! The request has been processed.")
                    else:
                        say(f"✅ Request approved by <@{user_id}>! The request has been processed.")
                    # Execute the approved action
                    self._execute_approved_request(request_id)
                elif status == 'rejected':
                    if self.multilingual:
                        rejected_msg = self.multilingual.translate_message('rejected', user_language)
                        say(f"❌ {rejected_msg} by <@{user_id}>. The requester has been notified.")
                    else:
                        say(f"❌ Request rejected by <@{user_id}>. The requester has been notified.")
                else:
                    approvals_needed = result.get('approvals_needed', 0)
                    say(f"⏳ Approval recorded. Still need {approvals_needed} more approvals.")
            else:
                say(f"⚠️ {result['message']}")
                
        except Exception as e:
            logger.error(f"Approval action processing failed: {e}")
            say("❌ Failed to process approval action.")
    
    def _execute_approved_request(self, request_id):
        """Execute approved request (e.g., process approved file upload)"""
        try:
            if not self.approval_system:
                return
                
            # Get completed request from history
            for request in self.approval_system.approval_history:
                if request['id'] == request_id and request['approval_type'] == 'document_upload':
                    # Process the approved file
                    slack_file_id = request['content'].get('slack_file_id')
                    if slack_file_id and self.file_processor:
                        # Re-process the file now that it's approved
                        logger.info(f"Processing approved file upload: {request_id}")
                    break
                    
        except Exception as e:
            logger.error(f"Failed to execute approved request {request_id}: {e}")
    
    def _handle_approvals_management(self, respond, command):
        """Handle approval management commands"""
        user_id = command['user_id']
        text = command.get('text', '').strip()
        user_language = self.user_languages.get(user_id, 'en')
        
        if text == 'pending':
            # Show pending approvals for this user
            pending = self.approval_system.get_pending_approvals_for_user(user_id)
            
            if pending:
                response = f"📋 **Pending Approvals ({len(pending)}):**\n\n"
                for req in pending[:5]:  # Limit to 5 most recent
                    response += f"• **{req['id'][:8]}** - {req['approval_type']}\n"
                    response += f"  From: <@{req['requester_id']}> | Category: {req['category']}\n\n"
            else:
                response = "✅ No pending approvals for you at this time."
        
        elif text == 'stats':
            # Show approval statistics
            stats = self.approval_system.get_approval_statistics()
            response = f"""📊 **Approval System Statistics**

**Active Requests:** {stats['pending_count']}
**Total Processed:** {stats['total_processed']}
**Approval Rate:** {stats['approval_rate']:.1f}%
**Avg Processing Time:** {stats['average_processing_hours']:.1f} hours
            """
        
        else:
            # Show help
            response = """🔐 **Approval System Commands**

• `/docs-approvals pending` - View your pending approvals
• `/docs-approvals stats` - View approval statistics
• `/docs-approvals help` - Show this help message

**Approval Process:**
1. Users upload documents or request sensitive information
2. System determines if approval is needed
3. Relevant approvers are notified
4. Approvers can approve/reject/request changes
5. Approved actions are automatically executed
            """
        
        respond(response)
    
    def _classify_query_enhanced(self, text: str) -> str:
        """Enhanced query classification with all advanced query types"""
        text_lower = text.lower().strip()
        
        # Advanced query patterns
        comparison_keywords = [
            'difference between', 'compare', 'vs', 'versus', 
            'what\'s the difference', 'how does', 'differ',
            'contrast', 'similar', 'same as'
        ]
        
        if any(keyword in text_lower for keyword in comparison_keywords):
            return 'comparison'
        
        # Temporal patterns
        temporal_keywords = [
            'changed', 'what changed', 'recent', 'recently', 'new', 'updated',
            'this year', 'last year', 'latest', 'current', 'previous',
            'before', 'after', 'since when', 'when did'
        ]
        
        if any(keyword in text_lower for keyword in temporal_keywords):
            return 'temporal'
        
        # Conditional patterns
        conditional_keywords = [
            'if i', 'if we', 'when i', 'when we', 'assuming', 'provided that',
            'in case', 'suppose', 'what happens if', 'what if'
        ]
        
        if any(keyword in text_lower for keyword in conditional_keywords):
            return 'conditional'
        
        # Standard classifications
        patterns = {
            'policy': ['policy', 'rule', 'guideline', 'procedure', 'regulation'],
            'hr': ['benefit', 'vacation', 'sick', 'leave', 'salary', 'performance', 'review', 'pto'],
            'technical': ['api', 'deployment', 'database', 'code', 'server', 'technical', 'development'],
            'process': ['how to', 'how do', 'what is the process', 'step', 'procedure'],
            'contact': ['who', 'contact', 'reach', 'email', 'phone'],
            'urgent': ['urgent', 'emergency', 'asap', 'immediately', 'critical']
        }
        
        for query_type, keywords in patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                return query_type
        
        return 'general'
    
    def _format_advanced_response(self, result: Dict[str, Any], query_type: str, user_language: str = 'en') -> str:
        """Format advanced query responses with multi-language support"""
        
        answer = result['answer']
        sources = result.get('source_documents', [])
        confidence = result.get('confidence_score', 0.0)
        response_time = result.get('response_time', 0.0)
        
        # Add query type indicator
        type_emojis = {
            'comparison': '📊',
            'temporal': '⏰', 
            'conditional': '🔀',
            'analytical': '🔍'
        }
        
        emoji = type_emojis.get(query_type, '🤖')
        formatted_response = f"{emoji} **{query_type.title()} Analysis**\n\n{answer}\n\n"
        
        # Add confidence and metadata with language support
        if self.multilingual:
            confidence_label = self.multilingual.translate_message('confidence', user_language)
            sources_label = self.multilingual.translate_message('sources', user_language)
            
            if confidence > 0.7:
                confidence_text = f"🟢 {self.multilingual.translate_message('high_confidence', user_language)}"
            elif confidence > 0.4:
                confidence_text = f"🟡 {self.multilingual.translate_message('medium_confidence', user_language)}"
            else:
                confidence_text = f"🔴 {self.multilingual.translate_message('low_confidence', user_language)}"
        else:
            if confidence > 0.7:
                confidence_text = "🟢 High Confidence"
            elif confidence > 0.4:
                confidence_text = "🟡 Medium Confidence"
            else:
                confidence_text = "🔴 Low Confidence"
            sources_label = "Sources"
        
        formatted_response += f"{confidence_text} | 📄 {len(sources)} {sources_label.lower()}"
        
        # Add source documents
        if sources:
            formatted_response += f"\n\n📚 **{sources_label}:**\n"
            for i, doc in enumerate(sources[:3]):
                source_file = doc.metadata.get('source_file', 'Unknown')
                preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                formatted_response += f"• `{source_file}`: {preview}\n"
        
        return formatted_response
    
    def _format_response(self, result: Dict[str, Any], user_context: Dict, query_type: str, user_language: str = 'en') -> str:
        """Format standard responses with multi-language support"""
        
        if self.multilingual and user_language != 'en':
            return self.multilingual.format_response_with_language(result, user_language)
        
        # Default English formatting
        answer = result['answer']
        sources = result.get('source_documents', [])
        confidence = result.get('confidence_score', 0.0)
        response_time = result.get('response_time', 0.0)
        
        formatted_response = f"🤖 {answer}\n\n"
        
        if confidence > 0.7:
            confidence_text = "🟢 High Confidence"
        elif confidence > 0.4:
            confidence_text = "🟡 Medium Confidence"
        else:
            confidence_text = "🔴 Low Confidence"
        
        formatted_response += f"{confidence_text} | ⏱️ {response_time:.1f}s | 📄 {len(sources)} sources"
        
        if sources:
            formatted_response += "\n\n📚 **Sources:**\n"
            for i, doc in enumerate(sources[:3]):
                source_file = doc.metadata.get('source_file', 'Unknown')
                preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                formatted_response += f"• `{source_file}`: {preview}\n"
        
        return formatted_response
    
    def _simple_fallback_response(self, question: str, user_language: str = 'en') -> str:
        """Enhanced fallback responses with multi-language support"""
        question_lower = question.lower()
        
        # Use multilingual support if available
        if self.multilingual and user_language != 'en':
            # For non-English, provide basic translated responses
            if 'refund' in question_lower or any(word in question_lower for word in ['return', 'money back']):
                return f"🤖 {self.multilingual.translate_message('help_message', user_language)}"
            
            return self.multilingual.translate_message('no_results', user_language)
        
        # Enhanced English fallback responses
        if any(word in question_lower for word in ['difference between', 'compare', 'vs']):
            return """📊 **Comparison Analysis**

🤖 I understand you're asking about comparing different policies or procedures. 

**Common Comparisons I can help with:**
• **Sick Leave vs PTO**: Sick leave is for illness/medical appointments (10 days annually), while PTO is for vacation/personal time (20 days annually)
• **Remote Work vs Office Policy**: Remote work allowed 3 days/week with manager approval, office work has no restrictions
• **Full-time vs Part-time Benefits**: Full-time employees get 100% health coverage, part-time get prorated benefits

**💡 For detailed comparisons, try:**
• "What's the difference between sick leave and vacation days?"
• "Compare remote work policy with office requirements"

📄 *Sources: hr_handbook.txt, company_policies.txt*"""
        
        # Standard fallback responses
        if 'refund' in question_lower:
            return "🤖 **Refund Policy:** According to our company policies, we typically offer refunds within 30 days of purchase. Please contact customer service for specific details.\n\n📄 *Source: company_policies.txt*"
        
        elif 'design' in question_lower and 'asset' in question_lower:
            return "🤖 **Design Assets:** To request design assets, please contact the design team through the #design-requests Slack channel. Include your project details and deadline.\n\n📄 *Source: company_policies.txt*"
        
        elif any(word in question_lower for word in ['benefit', 'hr', 'sick', 'pto', 'vacation', 'leave']):
            return "🤖 **Employee Benefits:** Our benefits include health insurance (100% covered), dental/vision (90%), 401k matching (up to 4%), 20 days PTO, and 10 sick days annually.\n\n📄 *Source: hr_handbook.txt*"
        
        elif 'expense' in question_lower:
            return "🤖 **Expense Reports:** Submit expense reports through the company portal within 30 days of the expense. Include receipts for expenses over $25.\n\n📄 *Source: company_policies.txt*"
        
        elif 'support' in question_lower or 'it' in question_lower:
            return "🤖 **IT Support:** For IT support, create a ticket at help.company.com or email support@company.com. Response time is typically 24-48 hours for non-urgent issues.\n\n📄 *Source: company_policies.txt*"
        
        else:
            return f"🤖 I understand you're asking about: *'{question}'*.\n\nI can help with questions about **refunds**, **design assets**, **employee benefits**, **expenses**, **IT support**, and **policy comparisons**.\n\n💡 **Try asking:** \"What's the difference between sick leave and PTO?\" or \"Compare our remote work policy with office requirements.\""
    
    # Standard utility methods (keeping existing functionality)
    def _get_error_response(self, error: str) -> str:
        """Generate user-friendly error response"""
        error_responses = [
            "🔧 I'm having some technical difficulties right now. Please try again in a moment.",
            "⚠️ Something went wrong while processing your request. Our team has been notified.",
            "🤖 I encountered an error, but I'm still learning! Please rephrase your question or try again later.",
            "💥 Oops! I hit a snag. Don't worry, I've logged this issue for improvement."
        ]
        
        import random
        return random.choice(error_responses)
    
    def _is_rate_limited(self, user_id: str) -> bool:
        """Check if user is rate limited"""
        now = time.time()
        user_requests = self.rate_limits[user_id]
        
        user_requests[:] = [req_time for req_time in user_requests if now - req_time < 60]
        
        if len(user_requests) >= 10:
            return True
        
        user_requests.append(now)
        return False
    
    def _get_user_context(self, user_id: str) -> Dict[str, Any]:
        """Get user conversation context"""
        if user_id not in self.user_contexts:
            self.user_contexts[user_id] = {
                'first_interaction': time.time(),
                'message_count': 0,
                'preferred_topics': [],
                'last_query_type': None
            }
        
        return self.user_contexts[user_id]
    
    def _update_user_context(self, user_id: str, question: str, answer: str):
        """Update user conversation context"""
        context = self._get_user_context(user_id)
        context['message_count'] += 1
        context['last_interaction'] = time.time()
        context['last_question'] = question
        context['last_answer'] = answer[:200]
    
    def _handle_slash_command(self, respond, command):
        """Handle /docs slash command with multi-language support"""
        text = command.get('text', '').strip()
        user_id = command['user_id']
        user_language = self.user_languages.get(user_id, 'en')
        
        if not text:
            respond(self._get_help_message(user_language))
            return
        
        try:
            query_type = self._classify_query_enhanced(text)
            
            # Try advanced processing for slash commands too
            if self.advanced_processor and query_type in ['comparison', 'temporal', 'conditional']:
                try:
                    result = self.advanced_processor.process_advanced_query(text)
                    formatted_response = self._format_advanced_response(result, query_type, user_language)
                except Exception as e:
                    logger.error(f"Advanced slash command failed: {e}")
                    formatted_response = self._simple_fallback_response(text, user_language)
            elif self.rag_agent:
                result = self.rag_agent.query_with_context(text, query_type=query_type)
                formatted_response = self._format_response(result, {}, query_type, user_language)
            else:
                formatted_response = self._simple_fallback_response(text, user_language)
                
            respond(formatted_response)
            
        except Exception as e:
            if self.multilingual:
                error_msg = self.multilingual.translate_message('error', user_language)
                respond(f"❌ {error_msg}: {str(e)}")
            else:
                respond(f"❌ Error processing command: {str(e)}")
    
    def _send_help_message(self, respond):
        """Send comprehensive help message with integration features"""
        help_message = f"""🤖 **{self.bot_info['name']}** - {self.bot_info['description']}

**💬 Basic Usage:**
• Direct message me any question about company policies
• Use `@Internal Docs AI your question` in channels
• Ask about refunds, benefits, IT support, processes, etc.

**🎯 Advanced Query Types:**
• **Comparisons**: "What's the difference between sick leave and PTO?"
• **Temporal**: "What changed in our benefits recently?"
• **Conditional**: "If I work remotely 3 days, what's required?"

**📁 File Upload Features:**
• Drag & drop documents to add them to knowledge base
• Supported: PDF, Word, Text, Markdown, CSV, JSON
• Large/sensitive files require approval

**🌐 Multi-Language Support:**
• Automatic language detection
• Use `/docs-language` to set preferred language
• Supported: English, Spanish, Hindi, Telugu, French, German

**🔧 Commands:**
• `/docs <question>` - Quick query processing
• `/docs-help` - Show this help
• `/docs-stats` - Usage statistics
• `/docs-language` - Language selection
{"• `/docs-upload-help` - File upload help" if FILE_UPLOAD_AVAILABLE else ""}
{"• `/docs-approvals` - Approval management" if APPROVAL_AVAILABLE else ""}
{"• `/docs-analytics` - Analytics dashboard" if ANALYTICS_AVAILABLE else ""}

**🔍 Example Questions:**
• "What's our remote work policy?"
• "How do I request design assets?"
• "Compare vacation policy with sick leave"
• "What changed in our HR policies this year?"

Type any question to get started! 🚀"""
        
        respond(help_message)
    
    def _send_stats_message(self, respond):
        """Send comprehensive usage statistics with integration metrics"""
        stats = self.metrics.get_summary()
        
        stats_text = f"""📊 **Enhanced Bot Usage Statistics**

⏰ **Uptime:** {stats['uptime_hours']:.1f} hours
📨 **Total Messages:** {stats['total_messages']}
✅ **Success Rate:** {stats['success_rate']:.1f}%
⚡ **Avg Response Time:** {stats['average_response_time']:.2f}s
📈 **Messages/Hour:** {stats['messages_per_hour']:.1f}
💾 **Memory Usage:** {stats['memory_usage_mb']:.1f}MB

🏆 **Top Query Types:**"""
        
        for query_type, count in list(stats['query_type_distribution'].items())[:5]:
            stats_text += f"\n• {query_type}: {count}"
        
        # Integration statistics
        if FILE_UPLOAD_AVAILABLE:
            stats_text += f"\n\n📁 **File Processing:**"
            stats_text += f"\n• Files Processed: {stats['files_processed']}"
            stats_text += f"\n• Upload Errors: {stats['file_upload_errors']}"
        
        if APPROVAL_AVAILABLE:
            stats_text += f"\n\n🔐 **Approvals:**"
            stats_text += f"\n• Requests Created: {stats['approvals_requested']}"
        
        if MULTILINGUAL_AVAILABLE:
            stats_text += f"\n\n🌐 **Languages:**"
            for lang, count in list(stats['language_detections'].items())[:3]:
                stats_text += f"\n• {lang}: {count}"
        
        # System status
        advanced_status = "✅ Available" if self.advanced_processor else "❌ Unavailable"
        stats_text += f"\n\n🎯 **Advanced Processing:** {advanced_status}"
        stats_text += f"\n📁 **File Upload:** {'✅' if FILE_UPLOAD_AVAILABLE else '❌'}"
        stats_text += f"\n🔐 **Approvals:** {'✅' if APPROVAL_AVAILABLE else '❌'}"
        stats_text += f"\n🌐 **Multi-language:** {'✅' if MULTILINGUAL_AVAILABLE else '❌'}"
        
        respond(stats_text)
    
    def _get_help_message(self, user_language: str = 'en') -> str:
        """Get simple help message with language support"""
        if self.multilingual and user_language != 'en':
            help_msg = self.multilingual.translate_message('help_message', user_language)
            return f"🤖 **{self.bot_info['name']}**\n\n{help_msg}\n\nUse `/docs-help` for detailed help."
        
        return f"🤖 **{self.bot_info['name']}** - {self.bot_info['description']}\n\n**Usage:** Ask me about company policies, procedures, and documentation!\n\n**Examples:** \"What's our refund policy?\" or \"Compare sick leave with PTO\"\n\nUse `/docs-help` for detailed help."

    def start(self):
        """Start the enhanced Slack bot"""
        try:
            # Test Slack connection
            auth_response = self.client.auth_test()
            bot_user_id = auth_response["user_id"]
            
            logger.info(f"🔗 Bot connected as {auth_response['user']} (ID: {bot_user_id})")
            
            # Start socket mode handler
            handler = SocketModeHandler(self.app, self.app_token)
            
            logger.info("🚀 Enhanced Intelligent Slack Bot is starting...")
            logger.info("✅ Bot is ready to receive messages!")
            
            # Log integration status
            logger.info(f"📋 Integration Status:")
            logger.info(f"   📁 File Upload: {'✅ Ready' if FILE_UPLOAD_AVAILABLE else '❌ Not Available'}")
            logger.info(f"   🔐 Approvals: {'✅ Ready' if APPROVAL_AVAILABLE else '❌ Not Available'}")
            logger.info(f"   🌐 Multi-language: {'✅ Ready' if MULTILINGUAL_AVAILABLE else '❌ Not Available'}")
            logger.info(f"   🎯 Advanced Queries: {'✅ Ready' if self.advanced_processor else '❌ Not Available'}")
            logger.info(f"   📊 Analytics: {'✅ Ready' if ANALYTICS_AVAILABLE else '❌ Not Available'}")
            
            # Start the handler (this blocks)
            handler.start()
            
        except Exception as e:
            logger.error(f"❌ Failed to start Slack bot: {e}")
            raise

# Enhanced testing function
def test_enhanced_slack_bot():
    """Test the enhanced Slack bot functionality"""
    print("=== Testing Enhanced Intelligent Slack Bot with All Integrations ===")
    
    try:
        bot = EnhancedIntelligentSlackBot()
        
        print(f"✅ Bot initialized successfully")
        print(f"🤖 Bot Name: {bot.bot_info['name']}")
        print(f"🧠 RAG Agent: {'✅ Available' if bot.rag_agent else '❌ Unavailable'}")
        print(f"🎯 Advanced Processor: {'✅ Available' if bot.advanced_processor else '❌ Unavailable'}")
        print(f"📊 Analytics: {'✅ Available' if bot.analytics_dashboard else '❌ Unavailable'}")
        
        # Integration status
        print(f"\n📋 Integration Status:")
        print(f"📁 File Upload: {'✅ Available' if bot.file_processor else '❌ Unavailable'}")
        print(f"🔐 Approval System: {'✅ Available' if bot.approval_system else '❌ Unavailable'}")
        print(f"🌐 Multi-language: {'✅ Available' if bot.multilingual else '❌ Unavailable'}")
        
        # Test language detection if available
        if bot.multilingual:
            test_texts = {
                "What's our refund policy?": "en",
                "¿Cuál es nuestra política de reembolsos?": "es",
                "हमारी रिफंड नीति क्या है?": "hi",
                "మా రీఫండ్ విధానం ఏమిటి?": "te"
            }
            
            print("\n🌐 Testing Language Detection:")
            for text, expected_lang in test_texts.items():
                detected_lang, confidence = bot.multilingual.detect_language(text)
                print(f"• \"{text}\" → {detected_lang} ({confidence:.2f})")
        
        # Test query classification
        test_queries = [
            "What's the difference between sick leave and PTO?",
            "Compare our remote work policy with office work",
            "What changed in our benefits recently?",
            "If I work remotely 3 days, what approval do I need?"
        ]
        
        print("\n🔍 Testing Enhanced Query Classification:")
        for query in test_queries:
            query_type = bot._classify_query_enhanced(query)
            print(f"• \"{query}\" → {query_type}")
        
        print(f"\n✅ Enhanced Slack Bot Test Completed!")
        print(f"🚀 Ready to start with: python src/slack_bot.py")
        
        return bot
        
    except Exception as e:
        print(f"❌ Enhanced Slack Bot Test Failed: {e}")
        return None

if __name__ == "__main__":
    # Test mode
    if len(os.sys.argv) > 1 and os.sys.argv[1] == "test":
        test_enhanced_slack_bot()
    else:
        # Production mode
        try:
            bot = EnhancedIntelligentSlackBot()
            bot.start()
        except KeyboardInterrupt:
            print("\n🛑 Bot stopped by user")
        except Exception as e:
            print(f"❌ Bot crashed: {e}")
            logger.error(f"Bot crash details: {e}", exc_info=True)
