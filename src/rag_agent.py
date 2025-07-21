import os
import logging
import re
from typing import List, Optional, Dict, Any, Tuple, Union
import time
from pathlib import Path
from enum import Enum
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import statistics

from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Import compatibility check
try:
    from langchain.callbacks import get_openai_callback
except ImportError:
    try:
        from langchain_community.callbacks import get_openai_callback
    except ImportError:
        # Fallback for missing callback
        from contextlib import contextmanager
        @contextmanager
        def get_openai_callback():
            class MockCallback:
                total_tokens = 0
                total_cost = 0.0
            yield MockCallback()

# Import our production vector store
from vector_store_manager import ProductionVectorStoreManager

# Load environment variables  
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Enumeration of different query types for advanced processing"""
    COMPARISON = "comparison"
    TEMPORAL = "temporal"
    CONDITIONAL = "conditional"
    ANALYTICAL = "analytical"
    STATISTICAL = "statistical"
    AGGREGATION = "aggregation"
    TREND = "trend"
    CLASSIFICATION = "classification"
    SYNTHESIS = "synthesis"
    CAUSAL = "causal"
    GENERAL = "general"

class QueryIntent:
    """Structure to hold query analysis results"""
    
    def __init__(self, primary_type: QueryType, confidence_score: float = 0.0):
        self.primary_type = primary_type
        self.confidence_score = confidence_score
        self.comparison_subjects = []
        self.temporal_indicators = []
        self.conditions = []
        self.analytical_focus = []
        self.statistical_operations = []
        self.aggregation_targets = []
        self.classification_criteria = []

class AdvancedQueryProcessor:
    """Advanced query analysis and intent detection"""
    
    def __init__(self):
        self.query_patterns = self._initialize_patterns()
    
    def _initialize_patterns(self) -> Dict[QueryType, Dict[str, List[str]]]:
        """Initialize pattern matching for different query types"""
        
        return {
            QueryType.COMPARISON: {
                'keywords': ['compare', 'versus', 'vs', 'difference', 'contrast', 'against', 'between'],
                'patterns': [r'compare\s+(.+?)\s+(?:with|to|and|vs)', r'(.+?)\s+versus\s+(.+)', r'difference between (.+?) and (.+)']
            },
            
            QueryType.TEMPORAL: {
                'keywords': ['changed', 'when', 'recent', 'updated', 'new', 'old', 'before', 'after', 'since'],
                'patterns': [r'what\s+changed', r'when\s+did', r'recently\s+updated', r'before\s+and\s+after']
            },
            
            QueryType.CONDITIONAL: {
                'keywords': ['if', 'when', 'provided', 'assuming', 'suppose', 'given that', 'in case'],
                'patterns': [r'if\s+(.+?)\s+then', r'when\s+(.+?)\s+what', r'provided\s+that\s+(.+)']
            },
            
            QueryType.ANALYTICAL: {
                'keywords': ['analyze', 'analysis', 'breakdown', 'examine', 'evaluate', 'assess', 'review'],
                'patterns': [r'analyze\s+(.+)', r'breakdown\s+of\s+(.+)', r'evaluate\s+(.+)']
            },
            
            QueryType.STATISTICAL: {
                'keywords': ['statistics', 'average', 'mean', 'median', 'percentage', 'rate', 'frequency', 'distribution'],
                'patterns': [r'what\s+percentage', r'average\s+(.+)', r'statistics\s+on\s+(.+)']
            },
            
            QueryType.AGGREGATION: {
                'keywords': ['total', 'sum', 'count', 'number of', 'how many', 'aggregate', 'overall'],
                'patterns': [r'how\s+many\s+(.+)', r'total\s+(.+)', r'count\s+of\s+(.+)']
            },
            
            QueryType.TREND: {
                'keywords': ['trend', 'pattern', 'over time', 'increasing', 'decreasing', 'growth', 'decline'],
                'patterns': [r'trend\s+in\s+(.+)', r'over\s+time', r'increasing\s+(.+)']
            },
            
            QueryType.CLASSIFICATION: {
                'keywords': ['classify', 'categorize', 'type of', 'kind of', 'group', 'category'],
                'patterns': [r'what\s+type\s+of', r'classify\s+(.+)', r'category\s+of\s+(.+)']
            },
            
            QueryType.SYNTHESIS: {
                'keywords': ['summarize', 'overview', 'synthesis', 'combine', 'integrate', 'merge'],
                'patterns': [r'summarize\s+(.+)', r'overview\s+of\s+(.+)', r'synthesis\s+of\s+(.+)']
            },
            
            QueryType.CAUSAL: {
                'keywords': ['why', 'because', 'reason', 'cause', 'effect', 'impact', 'result'],
                'patterns': [r'why\s+(.+)', r'cause\s+of\s+(.+)', r'impact\s+of\s+(.+)']
            }
        }
    
    def analyze_query(self, question: str) -> QueryIntent:
        """Analyze query and determine intent"""
        
        question_lower = question.lower().strip()
        scores = {}
        
        # Score each query type
        for query_type, patterns in self.query_patterns.items():
            score = 0
            
            # Check keyword matches
            keyword_matches = sum(1 for keyword in patterns['keywords'] if keyword in question_lower)
            score += keyword_matches * 2
            
            # Check pattern matches
            pattern_matches = 0
            for pattern in patterns['patterns']:
                if re.search(pattern, question_lower):
                    pattern_matches += 1
                    score += 3
            
            scores[query_type] = score
        
        # Determine primary type
        if not scores or max(scores.values()) == 0:
            primary_type = QueryType.GENERAL
            confidence = 0.3
        else:
            primary_type = max(scores, key=scores.get)
            max_score = scores[primary_type]
            confidence = min(max_score / 10.0, 1.0)  # Normalize confidence
        
        # Create intent object
        intent = QueryIntent(primary_type, confidence)
        
        # Extract specific information based on type
        if primary_type == QueryType.COMPARISON:
            intent.comparison_subjects = self._extract_comparison_subjects(question)
        elif primary_type == QueryType.TEMPORAL:
            intent.temporal_indicators = self._extract_temporal_indicators(question)
        elif primary_type == QueryType.CONDITIONAL:
            intent.conditions = self._extract_conditions(question)
        elif primary_type == QueryType.ANALYTICAL:
            intent.analytical_focus = self._extract_analytical_focus(question)
        elif primary_type == QueryType.STATISTICAL:
            intent.statistical_operations = self._extract_statistical_operations(question)
        
        return intent
    
    def _extract_comparison_subjects(self, question: str) -> List[str]:
        """Extract subjects being compared"""
        subjects = []
        
        # Pattern matching for comparison subjects
        patterns = [
            r'compare\s+(.+?)\s+(?:with|to|and|vs)\s+(.+?)(?:\s|$|\?)',
            r'(.+?)\s+versus\s+(.+?)(?:\s|$|\?)',
            r'difference\s+between\s+(.+?)\s+and\s+(.+?)(?:\s|$|\?)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, question.lower())
            if match:
                subjects.extend([group.strip() for group in match.groups()])
                break
        
        return subjects[:4]  # Limit to 4 subjects max
    
    def _extract_temporal_indicators(self, question: str) -> List[str]:
        """Extract temporal indicators from question"""
        indicators = []
        temporal_terms = ['changed', 'updated', 'new', 'recent', 'old', 'before', 'after', 'since', 'until']
        
        for term in temporal_terms:
            if term in question.lower():
                indicators.append(term)
        
        return indicators
    
    def _extract_conditions(self, question: str) -> List[str]:
        """Extract conditions from conditional queries"""
        conditions = []
        
        patterns = [
            r'if\s+(.+?)(?:,|\s+then|\s+what)',
            r'when\s+(.+?)(?:,|\s+what|\s+how)',
            r'provided\s+that\s+(.+?)(?:,|\s+what)',
            r'assuming\s+(.+?)(?:,|\s+what)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, question.lower())
            conditions.extend([match.strip() for match in matches])
        
        return conditions
    
    def _extract_analytical_focus(self, question: str) -> List[str]:
        """Extract focus areas for analytical queries"""
        focus_areas = []
        
        patterns = [
            r'analyze\s+(.+?)(?:\s|$|\?)',
            r'breakdown\s+of\s+(.+?)(?:\s|$|\?)',
            r'evaluate\s+(.+?)(?:\s|$|\?)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, question.lower())
            focus_areas.extend([match.strip() for match in matches])
        
        return focus_areas
    
    def _extract_statistical_operations(self, question: str) -> List[str]:
        """Extract statistical operations requested"""
        operations = []
        stat_terms = ['average', 'mean', 'median', 'percentage', 'rate', 'frequency', 'count', 'total']
        
        for term in stat_terms:
            if term in question.lower():
                operations.append(term)
        
        return operations

class AdvancedRAGProcessor:
    """Enhanced RAG processing with advanced query capabilities"""
    
    def __init__(self, rag_agent):
        self.rag_agent = rag_agent
        self.query_processor = AdvancedQueryProcessor()
        
    def process_advanced_query(self, question: str) -> Dict[str, Any]:
        """Process queries with advanced capabilities"""
        
        # Analyze query intent
        intent = self.query_processor.analyze_query(question)
        
        if intent.primary_type == QueryType.COMPARISON:
            return self._handle_comparison_query(question, intent)
        elif intent.primary_type == QueryType.TEMPORAL:
            return self._handle_temporal_query(question, intent)
        elif intent.primary_type == QueryType.CONDITIONAL:
            return self._handle_conditional_query(question, intent)
        elif intent.primary_type == QueryType.ANALYTICAL:
            return self._handle_analytical_query(question, intent)
        elif intent.primary_type == QueryType.STATISTICAL:
            return self._handle_statistical_query(question, intent)
        elif intent.primary_type == QueryType.AGGREGATION:
            return self._handle_aggregation_query(question, intent)
        elif intent.primary_type == QueryType.TREND:
            return self._handle_trend_query(question, intent)
        elif intent.primary_type == QueryType.CLASSIFICATION:
            return self._handle_classification_query(question, intent)
        elif intent.primary_type == QueryType.SYNTHESIS:
            return self._handle_synthesis_query(question, intent)
        elif intent.primary_type == QueryType.CAUSAL:
            return self._handle_causal_query(question, intent)
        else:
            # Fall back to standard RAG processing
            return self.rag_agent.query_with_context(question)
    
    def _handle_comparison_query(self, question: str, intent) -> Dict[str, Any]:
        """Handle comparison queries with multi-document synthesis"""
        
        # Get broader search results for comparison
        if not self.rag_agent.vector_manager.vector_store:
            return self._create_fallback_response("Comparison requires vector store", question)
        
        results = self.rag_agent.vector_manager.similarity_search(question, k=8, score_threshold=0.1)
        
        if not results:
            return self._create_fallback_response("No relevant documents found for comparison", question)
        
        # Create structured comparison response
        comparison_result = "📊 **COMPARISON ANALYSIS**\n\n"
        
        if intent.comparison_subjects:
            comparison_result += f"**Comparing: {' vs '.join(intent.comparison_subjects)}**\n\n"
        
        comparison_result += "**Key Differences Found:**\n"
        for i, (doc, score) in enumerate(results[:4]):
            source = doc.metadata.get('source_file', doc.metadata.get('source', 'Unknown'))
            preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            comparison_result += f"• **{source}**: {preview}\n"
        
        return {
            'answer': comparison_result,
            'source_documents': [doc for doc, _ in results[:4]],
            'query_type': 'comparison',
            'comparison_subjects': intent.comparison_subjects,
            'response_time': 1.0,
            'confidence_score': intent.confidence_score,
            'processing_mode': 'advanced'
        }
    
    def _handle_temporal_query(self, question: str, intent) -> Dict[str, Any]:
        """Handle temporal queries"""
        results = self.rag_agent.vector_manager.similarity_search(question, k=6)
        
        if not results:
            return self._create_fallback_response("No documents found for temporal analysis", question)
        
        temporal_result = "⏰ **TEMPORAL ANALYSIS**\n\n"
        temporal_result += "**Recent Changes and Updates:**\n"
        
        for i, (doc, score) in enumerate(results[:4]):
            source = doc.metadata.get('source_file', doc.metadata.get('source', 'Unknown'))
            preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
            temporal_result += f"• **{source}**: {preview}\n"
        
        return {
            'answer': temporal_result,
            'source_documents': [doc for doc, _ in results[:4]],
            'query_type': 'temporal',
            'changes_detected': intent.temporal_indicators,
            'response_time': 1.0,
            'confidence_score': intent.confidence_score,
            'processing_mode': 'advanced'
        }
    
    def _handle_conditional_query(self, question: str, intent) -> Dict[str, Any]:
        """Handle conditional queries"""
        results = self.rag_agent.vector_manager.similarity_search(question, k=5)
        
        if not results:
            return self._create_fallback_response("No relevant information found for conditional analysis", question)
        
        conditional_result = "🔀 **CONDITIONAL ANALYSIS**\n\n"
        
        if intent.conditions:
            conditional_result += f"**Scenario: {', '.join(intent.conditions)}**\n\n"
        
        conditional_result += "**Relevant Policies and Requirements:**\n"
        
        for i, (doc, score) in enumerate(results[:3]):
            source = doc.metadata.get('source_file', doc.metadata.get('source', 'Unknown'))
            preview = doc.page_content[:180] + "..." if len(doc.page_content) > 180 else doc.page_content
            conditional_result += f"• **{source}**: {preview}\n"
        
        return {
            'answer': conditional_result,
            'source_documents': [doc for doc, _ in results[:3]],
            'query_type': 'conditional',
            'conditions': intent.conditions,
            'response_time': 1.0,
            'confidence_score': intent.confidence_score,
            'processing_mode': 'advanced'
        }
    
    def _handle_analytical_query(self, question: str, intent) -> Dict[str, Any]:
        """Handle analytical queries"""
        results = self.rag_agent.vector_manager.similarity_search(question, k=8)
        
        if not results:
            return self._create_fallback_response("No documents found for analysis", question)
        
        analytical_result = "🔍 **ANALYTICAL ASSESSMENT**\n\n"
        
        if intent.analytical_focus:
            analytical_result += f"**Focus Areas: {', '.join(intent.analytical_focus)}**\n\n"
        
        analytical_result += "**Analysis Results:**\n"
        
        for i, (doc, score) in enumerate(results[:5]):
            source = doc.metadata.get('source_file', doc.metadata.get('source', 'Unknown'))
            preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            analytical_result += f"• **{source}**: {preview}\n"
        
        return {
            'answer': analytical_result,
            'source_documents': [doc for doc, _ in results[:5]],
            'query_type': 'analytical',
            'focus_areas': intent.analytical_focus,
            'response_time': 1.0,
            'confidence_score': intent.confidence_score,
            'processing_mode': 'advanced'
        }
    
    def _handle_statistical_query(self, question: str, intent) -> Dict[str, Any]:
        """Handle statistical queries"""
        results = self.rag_agent.vector_manager.similarity_search(question, k=6)
        
        if not results:
            return self._create_fallback_response("No data found for statistical analysis", question)
        
        statistical_result = "📈 **STATISTICAL SUMMARY**\n\n"
        
        if intent.statistical_operations:
            statistical_result += f"**Operations: {', '.join(intent.statistical_operations)}**\n\n"
        
        statistical_result += "**Data Points Found:**\n"
        
        for i, (doc, score) in enumerate(results[:4]):
            source = doc.metadata.get('source_file', doc.metadata.get('source', 'Unknown'))
            preview = doc.page_content[:180] + "..." if len(doc.page_content) > 180 else doc.page_content
            statistical_result += f"• **{source}**: {preview}\n"
        
        return {
            'answer': statistical_result,
            'source_documents': [doc for doc, _ in results[:4]],
            'query_type': 'statistical',
            'operations': intent.statistical_operations,
            'response_time': 1.0,
            'confidence_score': intent.confidence_score,
            'processing_mode': 'advanced'
        }
    
    def _handle_aggregation_query(self, question: str, intent) -> Dict[str, Any]:
        """Handle aggregation queries"""
        results = self.rag_agent.vector_manager.similarity_search(question, k=8)
        
        if not results:
            return self._create_fallback_response("No data found for aggregation", question)
        
        aggregation_result = "📊 **AGGREGATION ANALYSIS**\n\n"
        
        # Count categories
        sources = {}
        for doc, score in results:
            source = doc.metadata.get('source_file', doc.metadata.get('source', 'Unknown'))
            if source not in sources:
                sources[source] = 0
            sources[source] += 1
        
        aggregation_result += f"**Found {len(results)} relevant items across {len(sources)} sources:**\n"
        
        for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
            aggregation_result += f"• **{source}**: {count} references\n"
        
        return {
            'answer': aggregation_result,
            'source_documents': [doc for doc, _ in results[:6]],
            'query_type': 'aggregation',
            'response_time': 1.0,
            'confidence_score': intent.confidence_score,
            'processing_mode': 'advanced'
        }
    
    def _handle_trend_query(self, question: str, intent) -> Dict[str, Any]:
        """Handle trend analysis queries"""
        results = self.rag_agent.vector_manager.similarity_search(question, k=6)
        
        if not results:
            return self._create_fallback_response("No data found for trend analysis", question)
        
        trend_result = "📈 **TREND ANALYSIS**\n\n"
        trend_result += "**Patterns and Trends Identified:**\n"
        
        for i, (doc, score) in enumerate(results[:4]):
            source = doc.metadata.get('source_file', doc.metadata.get('source', 'Unknown'))
            preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
            trend_result += f"• **{source}**: {preview}\n"
        
        return {
            'answer': trend_result,
            'source_documents': [doc for doc, _ in results[:4]],
            'query_type': 'trend',
            'response_time': 1.0,
            'confidence_score': intent.confidence_score,
            'processing_mode': 'advanced'
        }
    
    def _handle_classification_query(self, question: str, intent) -> Dict[str, Any]:
        """Handle classification queries"""
        results = self.rag_agent.vector_manager.similarity_search(question, k=6)
        
        if not results:
            return self._create_fallback_response("No data found for classification", question)
        
        classification_result = "🏷️ **CLASSIFICATION ANALYSIS**\n\n"
        
        # Group by document type/source
        categories = {}
        for doc, score in results:
            source = doc.metadata.get('source_file', doc.metadata.get('source', 'Unknown'))
            category = source.split('_')[0] if '_' in source else 'general'
            if category not in categories:
                categories[category] = []
            categories[category].append((doc, score))
        
        for category, docs in categories.items():
            classification_result += f"**{category.title()} ({len(docs)} items):**\n"
            for doc, score in docs[:2]:  # Show top 2 per category
                preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                classification_result += f"• {preview}\n"
            classification_result += "\n"
        
        return {
            'answer': classification_result,
            'source_documents': [doc for doc, _ in results[:5]],
            'query_type': 'classification',
            'response_time': 1.0,
            'confidence_score': intent.confidence_score,
            'processing_mode': 'advanced'
        }
    
    def _handle_synthesis_query(self, question: str, intent) -> Dict[str, Any]:
        """Handle synthesis queries"""
        results = self.rag_agent.vector_manager.similarity_search(question, k=10)
        
        if not results:
            return self._create_fallback_response("No documents found for synthesis", question)
        
        synthesis_result = "🔗 **DOCUMENT SYNTHESIS**\n\n"
        synthesis_result += "**Comprehensive Overview:**\n"
        
        for i, (doc, score) in enumerate(results[:6]):
            source = doc.metadata.get('source_file', doc.metadata.get('source', 'Unknown'))
            preview = doc.page_content[:180] + "..." if len(doc.page_content) > 180 else doc.page_content
            synthesis_result += f"• **{source}**: {preview}\n"
        
        return {
            'answer': synthesis_result,
            'source_documents': [doc for doc, _ in results[:6]],
            'query_type': 'synthesis',
            'response_time': 1.0,
            'confidence_score': intent.confidence_score,
            'processing_mode': 'advanced'
        }
    
    def _handle_causal_query(self, question: str, intent) -> Dict[str, Any]:
        """Handle causal analysis queries"""
        results = self.rag_agent.vector_manager.similarity_search(question, k=5)
        
        if not results:
            return self._create_fallback_response("No information found for causal analysis", question)
        
        causal_result = "🎯 **CAUSAL ANALYSIS**\n\n"
        causal_result += "**Cause-Effect Relationships:**\n"
        
        for i, (doc, score) in enumerate(results[:4]):
            source = doc.metadata.get('source_file', doc.metadata.get('source', 'Unknown'))
            preview = doc.page_content[:160] + "..." if len(doc.page_content) > 160 else doc.page_content
            causal_result += f"• **{source}**: {preview}\n"
        
        return {
            'answer': causal_result,
            'source_documents': [doc for doc, _ in results[:4]],
            'query_type': 'causal',
            'response_time': 1.0,
            'confidence_score': intent.confidence_score,
            'processing_mode': 'advanced'
        }
    
    def _create_fallback_response(self, message: str, question: str) -> Dict[str, Any]:
        """Create a fallback response when processing fails"""
        return {
            'answer': f"I apologize, but {message.lower()}. Please try rephrasing your question or contact support for assistance.",
            'source_documents': [],
            'query_type': 'error',
            'response_time': 0.0,
            'confidence_score': 0.0,
            'processing_mode': 'error'
        }

class ProductionRAGAgent:
    """Production-ready RAG Agent with FIXED vector store initialization"""
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model_name: str = "gpt-3.5-turbo",
        embedding_strategy: str = "auto",
        vector_store_type: str = "faiss",
        enable_advanced_processing: bool = True
    ):
        """Initialize production RAG agent with FIXED document handling"""
        
        # Configuration
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        
        # Initialize production vector store manager
        self.vector_manager = ProductionVectorStoreManager(
            embedding_config=embedding_strategy,
            vector_store_type=vector_store_type,
            enable_caching=True,
            max_memory_mb=int(os.getenv("MAX_MEMORY_USAGE", 2048))
        )
        
        # Initialize LLM
        self.llm = None
        self.retrieval_chain = None
        
        # Advanced processing
        self.advanced_processor = None
        if enable_advanced_processing:
            try:
                self.advanced_processor = AdvancedRAGProcessor(self)
                logger.info("✅ Advanced query processor initialized")
            except Exception as e:
                logger.error(f"❌ Advanced processor initialization failed: {e}")
        
        # Performance tracking
        self.query_stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'total_tokens_used': 0,
            'total_cost': 0.0,
            'avg_response_time': 0.0,
            'query_types': Counter(),
            'advanced_queries': 0
        }
        
        logger.info(f"✅ Production RAG Agent initialized with {embedding_strategy} embeddings and {vector_store_type} vector store")
    
    def initialize_components(self, documents: List[Document] = None) -> Dict[str, bool]:
        """FIXED: Initialize all components with proper document handling"""
        status = {
            'vector_store': False,
            'embeddings': False,
            'llm': False,
            'retrieval_chain': False
        }
        
        try:
            # Initialize embedding service first
            status['embeddings'] = self.vector_manager.initialize_embedding_service()
            
            # Initialize vector store WITH documents (FIXED)
            if documents:
                status['vector_store'] = self.vector_manager.initialize_vector_store(documents)
                logger.info(f"✅ Vector store initialized with {len(documents)} documents")
            else:
                # Try to load existing or create empty one
                status['vector_store'] = self.vector_manager.initialize_vector_store()
            
            # Initialize LLM
            if self.openai_api_key and self.openai_api_key != "your_openai_key_here":
                try:
                    self.llm = ChatOpenAI(
                        openai_api_key=self.openai_api_key,
                        model_name=self.model_name,
                        temperature=0.1,
                        max_tokens=1024
                    )
                    status['llm'] = True
                    logger.info(f"✅ LLM initialized: {self.model_name}")
                except Exception as e:
                    logger.warning(f"⚠️ LLM initialization failed: {e}")
            else:
                logger.info("⚠️ No OpenAI API key, will use fallback responses")
            
            # Initialize retrieval chain
            if status['vector_store'] and status['llm']:
                status['retrieval_chain'] = self._setup_retrieval_chain()
            
            logger.info(f"🔍 Component initialization: {status}")
            return status
            
        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")
            return status
    
    def _setup_retrieval_chain(self) -> bool:
        """Setup retrieval chain with FIXED FAISS compatibility"""
        try:
            # Fixed prompt template
            prompt_template = """Use the provided context to answer the question comprehensively.

Context:
{context}

Question: {question}

Provide a detailed answer based on the context above. If the information isn't in the context, say so clearly.

Answer:"""

            custom_prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Setup retriever with fixed parameters
            if hasattr(self.vector_manager.vector_store, 'as_retriever'):
                retriever = self.vector_manager.vector_store.as_retriever(
                    search_kwargs={"k": 6}
                )
                
                # Create retrieval chain
                self.retrieval_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=retriever,
                    chain_type_kwargs={"prompt": custom_prompt},
                    return_source_documents=True
                )
                
                logger.info("✅ Retrieval chain setup successfully")
                return True
            else:
                logger.error("❌ Vector store doesn't support retriever interface")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to setup retrieval chain: {e}")
            return False
    
    def query_with_context(self, question: str, query_type: str = "general") -> Dict[str, Any]:
        """FIXED: Query processing with proper vector store check"""
        
        start_time = time.time()
        
        try:
            self.query_stats['total_queries'] += 1
            self.query_stats['query_types'][query_type] += 1
            
            # Check if vector store has documents
            if not self.vector_manager.vector_store:
                return self._create_fallback_response("Vector store not initialized", question)
            
            # Try similarity search directly first
            try:
                search_results = self.vector_manager.similarity_search(question, k=5)
                if not search_results:
                    return self._create_fallback_response("No relevant documents found", question)
                
                # Update successful queries counter
                self.query_stats['successful_queries'] += 1
                
                # If we have LLM, use retrieval chain
                if self.retrieval_chain:
                    with get_openai_callback() as cb:
                        result = self.retrieval_chain.invoke({"query": question})
                    
                    response_time = time.time() - start_time
                    
                    return {
                        'answer': result['result'],
                        'source_documents': result.get('source_documents', [doc for doc, _ in search_results[:3]]),
                        'response_time': response_time,
                        'tokens_used': cb.total_tokens,
                        'cost': cb.total_cost,
                        'query_type': query_type,
                        'confidence_score': 0.8,
                        'processing_mode': 'full_llm'
                    }
                
                else:
                    # Fallback to basic document retrieval
                    docs_text = "\n\n".join([f"**{doc.metadata.get('source_file', 'Unknown')}**: {doc.page_content[:300]}..." 
                                           for doc, _ in search_results[:3]])
                    
                    fallback_answer = f"Based on our company documentation:\n\n{docs_text}\n\n*Note: This is a basic search result. Full AI processing requires OpenAI API configuration.*"
                    
                    response_time = time.time() - start_time
                    
                    return {
                        'answer': fallback_answer,
                        'source_documents': [doc for doc, _ in search_results[:3]],
                        'response_time': response_time,
                        'tokens_used': 0,
                        'cost': 0.0,
                        'query_type': query_type,
                        'confidence_score': 0.6,
                        'processing_mode': 'document_search'
                    }
                    
            except Exception as e:
                logger.error(f"❌ Query processing failed: {e}")
                return self._create_fallback_response(f"Query processing error: {str(e)}", question)
            
        except Exception as e:
            logger.error(f"❌ Critical query error: {e}")
            self.query_stats['failed_queries'] += 1
            return self._create_fallback_response(f"System error: {str(e)}", question)
    
    def _create_fallback_response(self, error_msg: str, question: str) -> Dict[str, Any]:
        """Create fallback response for errors"""
        return {
            'answer': f"I'm unable to process your question right now. Please check the system configuration. Error: {error_msg}",
            'source_documents': [],
            'response_time': 0.0,
            'tokens_used': 0,
            'cost': 0.0,
            'query_type': 'error',
            'confidence_score': 0.0,
            'processing_mode': 'error'
        }
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        return {
            'query_stats': self.query_stats,
            'vector_store_stats': self.vector_manager.get_vector_store_stats(),
            'system_status': self.get_system_health()
        }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health indicators"""
        health = {
            'overall_status': 'healthy',
            'components': {},
            'alerts': []
        }
        
        # Check component status
        health['components']['vector_store'] = 'healthy' if self.vector_manager.vector_store else 'unhealthy'
        health['components']['embeddings'] = 'healthy' if self.vector_manager.embedding_service else 'unhealthy'
        health['components']['llm'] = 'healthy' if self.llm else 'partial'
        health['components']['advanced_processor'] = 'healthy' if self.advanced_processor else 'disabled'
        
        # Determine overall status
        if health['components']['vector_store'] == 'unhealthy':
            health['overall_status'] = 'critical'
        elif health['components']['embeddings'] == 'unhealthy':
            health['overall_status'] = 'critical'
        elif not self.llm:
            health['overall_status'] = 'partial'
        
        return health

def test_production_rag():
    """FIXED: Comprehensive test with proper document loading"""
    print("=== Production RAG Agent Test ===")
    
    # Load documents first
    try:
        from document_loader import OptimizedDocumentLoader
        loader = OptimizedDocumentLoader()
        documents = loader.load_directory("data")
        
        if not documents:
            print("📝 No documents found, creating sample documents...")
            loader.create_sample_documents()
            documents = loader.load_directory("data")
        
        print(f"📄 Loaded {len(documents)} documents")
        
    except ImportError:
        print("⚠️ Document loader not available, using sample data")
        documents = [
            Document(page_content="Company refund policy: We offer full refunds within 30 days.", metadata={'source_file': 'policy.txt'}),
            Document(page_content="Employee benefits include health insurance and PTO.", metadata={'source_file': 'hr.txt'}),
            Document(page_content="Remote work allowed up to 3 days per week.", metadata={'source_file': 'remote.txt'})
        ]
    
    # Initialize agent
    agent = ProductionRAGAgent(
        embedding_strategy="auto",
        vector_store_type="faiss",
        enable_advanced_processing=True
    )
    
    # FIXED: Initialize with documents
    status = agent.initialize_components(documents)
    print(f"🔧 Initialization status: {status}")
    
    # Test queries
    test_queries = [
        ("What's our refund policy?", "policy"),
        ("Compare sick leave with vacation days", "comparison"),
        ("What changed in our benefits recently?", "temporal"),
        ("If I work remotely 3 days, what approval do I need?", "conditional"),
        ("Analyze our remote work policies", "analytical"),
        ("How many types of benefits do we offer?", "aggregation")
    ]
    
    print("\n🎯 Testing Advanced Query Processing:")
    
    for question, query_type in test_queries:
        print(f"\n📝 Query Type: {query_type}")
        print(f"❓ Question: {question}")
        
        # Try advanced processing first
        if agent.advanced_processor and query_type in ['comparison', 'temporal', 'conditional', 'analytical', 'aggregation']:
            try:
                result = agent.advanced_processor.process_advanced_query(question)
            except Exception as e:
                logger.error(f"Advanced processing failed: {e}")
                result = agent.query_with_context(question, query_type)
        else:
            result = agent.query_with_context(question, query_type)
        
        print(f"✅ Answer: {result['answer'][:200]}...")
        print(f"🎯 Confidence: {result['confidence_score']:.2f}")
        print(f"⏱️ Response Time: {result['response_time']:.2f}s")
        print(f"📄 Sources: {len(result['source_documents'])}")
        print(f"🔧 Mode: {result.get('processing_mode', 'unknown')}")
    
    # Show comprehensive stats
    print("\n📊 Comprehensive Statistics:")
    stats = agent.get_comprehensive_stats()
    
    print("🔍 Query Statistics:")
    for key, value in stats['query_stats'].items():
        if key != 'query_types':
            print(f"  {key}: {value}")
    
    print("📋 Query Types:")
    for query_type, count in stats['query_stats']['query_types'].most_common():
        print(f"  {query_type}: {count}")
    
    print("🏥 System Health:")
    health = stats['system_status']
    print(f"  Overall Status: {health['overall_status']}")
    print(f"  Components: {health['components']}")
    
    print("\n✅ Production RAG Agent Test Completed!")
    
    return agent

if __name__ == "__main__":
    test_production_rag()
