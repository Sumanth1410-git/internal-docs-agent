"""
Comprehensive tests for Multilingual Support System
Tests language detection, translation, and localization features
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

class TestMultilingualSupport:
    """Test suite for Multilingual Support System"""
    
    def test_multilingual_initialization(self):
        """Test multilingual support initialization"""
        from multilingual_support import MultilingualSupport
        
        ml_support = MultilingualSupport()
        
        assert ml_support is not None
        assert len(ml_support.supported_languages) >= 6
        assert 'en' in ml_support.supported_languages
        assert 'es' in ml_support.supported_languages
        assert 'hi' in ml_support.supported_languages
        assert 'te' in ml_support.supported_languages
    
    def test_language_detection_english(self):
        """Test English language detection"""
        from multilingual_support import MultilingualSupport
        
        ml_support = MultilingualSupport()
        
        # Test clear English text
        english_text = "What is our company refund policy for customers?"
        detected_lang, confidence = ml_support.detect_language(english_text)
        
        assert detected_lang == 'en'
        assert confidence > 0.0
    
    def test_language_detection_spanish(self):
        """Test Spanish language detection"""
        from multilingual_support import MultilingualSupport
        
        ml_support = MultilingualSupport()
        
        # Test Spanish text with clear indicators
        spanish_text = "¿Cuál es nuestra política de reembolsos para clientes?"
        detected_lang, confidence = ml_support.detect_language(spanish_text)
        
        assert detected_lang == 'es'
        assert confidence > 0.5  # Should be reasonably confident
    
    def test_language_detection_hindi(self):
        """Test Hindi language detection"""
        from multilingual_support import MultilingualSupport
        
        ml_support = MultilingualSupport()
        
        # Test Hindi text with Devanagari script
        hindi_text = "हमारी कंपनी की रिफंड नीति क्या है?"
        detected_lang, confidence = ml_support.detect_language(hindi_text)
        
        assert detected_lang == 'hi'
        assert confidence > 0.7  # Script-based detection should be highly confident
    
    def test_language_detection_telugu(self):
        """Test Telugu language detection (for your location in Telangana)"""
        from multilingual_support import MultilingualSupport
        
        ml_support = MultilingualSupport()
        
        # Test Telugu text with Telugu script
        telugu_text = "మా కంపెనీ రీఫండ్ విధానం ఏమిటి?"
        detected_lang, confidence = ml_support.detect_language(telugu_text)
        
        assert detected_lang == 'te'
        assert confidence > 0.7  # Script-based detection should be highly confident
    
    def test_language_detection_empty_text(self):
        """Test language detection with empty or minimal text"""
        from multilingual_support import MultilingualSupport
        
        ml_support = MultilingualSupport()
        
        # Empty text
        detected_lang, confidence = ml_support.detect_language("")
        assert detected_lang == 'en'  # Default fallback
        assert confidence <= 0.5
        
        # Very short text
        detected_lang, confidence = ml_support.detect_language("Hi")
        assert detected_lang == 'en'
        assert confidence <= 0.5
    
    def test_message_translation_basic(self):
        """Test basic message translation functionality"""
        from multilingual_support import MultilingualSupport
        
        ml_support = MultilingualSupport()
        
        # Test welcome message in different languages
        english_welcome = ml_support.translate_message('welcome', 'en')
        spanish_welcome = ml_support.translate_message('welcome', 'es')
        hindi_welcome = ml_support.translate_message('welcome', 'hi')
        
        assert 'Welcome' in english_welcome
        assert 'Bienvenido' in spanish_welcome
        assert 'स्वागत' in hindi_welcome
        
        # All should be different
        assert english_welcome != spanish_welcome
        assert english_welcome != hindi_welcome
    
    def test_message_translation_with_fallback(self):
        """Test message translation with fallback to English"""
        from multilingual_support import MultilingualSupport
        
        ml_support = MultilingualSupport()
        
        # Test unsupported language code
        unsupported_message = ml_support.translate_message('welcome', 'xx')
        english_message = ml_support.translate_message('welcome', 'en')
        
        assert unsupported_message == english_message  # Should fallback to English
    
    def test_missing_translation_key(self):
        """Test handling of missing translation keys"""
        from multilingual_support import MultilingualSupport
        
        ml_support = MultilingualSupport()
        
        # Test non-existent key
        missing
