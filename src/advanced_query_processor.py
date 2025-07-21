import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class QueryType(Enum):
    COMPARISON = "comparison"
    TEMPORAL = "temporal" 
    CONDITIONAL = "conditional"
    ANALYTICAL = "analytical"
    PROCEDURAL = "procedural"
    FACTUAL = "factual"

@dataclass
class QueryIntent:
    """Structured representation of user query intent"""
    primary_type: QueryType
    entities: List[str]
    conditions: List[str]
    temporal_indicators: List[str]
    comparison_subjects: List[str]
    confidence_score: float

class AdvancedQueryProcessor:
    """Intelligent query processing with intent recognition"""
    
    def __init__(self):
        self.comparison_patterns = [
            r"compare\s+(.+?)\s+(?:with|and|vs|versus)\s+(.+)",
            r"difference\s+between\s+(.+?)\s+and\s+(.+)",
            r"(.+?)\s+vs\s+(.+)",
            r"how\s+does\s+(.+?)\s+differ\s+from\s+(.+)"
        ]
        
        self.temporal_patterns = [
            r"what\s+changed?\s*(?:in|since|from)\s+(.+)",
            r"(?:recent|new|latest|updated?)\s+(.+)",
            r"(?:before|after|since)\s+(.+)",
            r"(?:this|last)\s+(?:year|month|quarter)\s+(.+)"
        ]
        
        self.conditional_patterns = [
            r"if\s+(.+?),?\s+(?:what|how|when|where)\s+(.+)",
            r"when\s+(.+?),?\s+(?:what|how)\s+(.+)",
            r"assuming\s+(.+?),?\s+(.+)",
            r"provided\s+that\s+(.+?),?\s+(.+)"
        ]
        
        self.entity_keywords = {
            'policy': ['policy', 'rule', 'guideline', 'regulation', 'procedure'],
            'benefits': ['benefit', 'insurance', 'health', 'dental', '401k', 'pto', 'vacation'],
            'process': ['process', 'workflow', 'step', 'procedure', 'how to'],
            'contact': ['contact', 'email', 'phone', 'reach', 'support'],
            'technical': ['api', 'system', 'server', 'database', 'code', 'deployment']
        }
    
    def analyze_query(self, query: str) -> QueryIntent:
        """Analyze query and extract structured intent"""
        
        query_lower = query.lower().strip()
        
        # Detect query type
        query_type = self._detect_query_type(query_lower)
        
        # Extract entities
        entities = self._extract_entities(query_lower)
        
        # Extract specific components based on type
        conditions = []
        temporal_indicators = []
        comparison_subjects = []
        
        if query_type == QueryType.COMPARISON:
            comparison_subjects = self._extract_comparison_subjects(query_lower)
        elif query_type == QueryType.TEMPORAL:
            temporal_indicators = self._extract_temporal_indicators(query_lower)
        elif query_type == QueryType.CONDITIONAL:
            conditions = self._extract_conditions(query_lower)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence(query_type, entities, query_lower)
        
        return QueryIntent(
            primary_type=query_type,
            entities=entities,
            conditions=conditions,
            temporal_indicators=temporal_indicators,
            comparison_subjects=comparison_subjects,
            confidence_score=confidence_score
        )
    
    def _detect_query_type(self, query: str) -> QueryType:
        """Detect the primary query type"""
        
        # Check for comparison patterns
        for pattern in self.comparison_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryType.COMPARISON
        
        # Check for temporal patterns
        for pattern in self.temporal_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryType.TEMPORAL
        
        # Check for conditional patterns
        for pattern in self.conditional_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryType.CONDITIONAL
        
        # Check for analytical indicators
        analytical_keywords = ['analyze', 'breakdown', 'explain why', 'how does', 'what causes']
        if any(keyword in query for keyword in analytical_keywords):
            return QueryType.ANALYTICAL
        
        # Check for procedural indicators
        procedural_keywords = ['how to', 'steps to', 'process for', 'way to']
        if any(keyword in query for keyword in procedural_keywords):
            return QueryType.PROCEDURAL
        
        return QueryType.FACTUAL
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract relevant entities from the query"""
        entities = []
        
        for category, keywords in self.entity_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    entities.append(category)
                    break
        
        # Extract quoted entities
        quoted_entities = re.findall(r'"([^"]*)"', query)
        entities.extend(quoted_entities)
        
        return list(set(entities))
    
    def _extract_comparison_subjects(self, query: str) -> List[str]:
        """Extract subjects being compared"""
        subjects = []
        
        for pattern in self.comparison_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                subjects.extend([group.strip() for group in match.groups()])
                break
        
        return subjects
    
    def _extract_temporal_indicators(self, query: str) -> List[str]:
        """Extract temporal indicators"""
        indicators = []
        
        temporal_terms = [
            'recently', 'lately', 'this year', 'last year', 'this month', 'last month',
            'new', 'updated', 'changed', 'modified', 'revised', 'current', 'previous'
        ]
        
        for term in temporal_terms:
            if term in query:
                indicators.append(term)
        
        return indicators
    
    def _extract_conditions(self, query: str) -> List[str]:
        """Extract conditional statements"""
        conditions = []
        
        for pattern in self.conditional_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                conditions.append(match.group(1).strip())
        
        return conditions
    
    def _calculate_confidence(self, query_type: QueryType, entities: List[str], query: str) -> float:
        """Calculate confidence score for query analysis"""
        
        base_confidence = 0.5
        
        # Boost confidence based on clear patterns
        if query_type == QueryType.COMPARISON and any(word in query for word in ['compare', 'vs', 'difference']):
            base_confidence += 0.3
        
        if query_type == QueryType.TEMPORAL and any(word in query for word in ['changed', 'new', 'updated']):
            base_confidence += 0.3
        
        if query_type == QueryType.CONDITIONAL and any(word in query for word in ['if', 'when', 'assuming']):
            base_confidence += 0.3
        
        # Boost confidence based on entities found
        if entities:
            base_confidence += min(len(entities) * 0.1, 0.2)
        
        return min(base_confidence, 1.0)
