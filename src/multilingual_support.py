import json
import re
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)

class MultilingualSupport:
    """Comprehensive multi-language support system"""
    
    def __init__(self):
        self.supported_languages = {
            'en': {'name': 'English', 'native': 'English'},
            'es': {'name': 'Spanish', 'native': 'Español'},  
            'fr': {'name': 'French', 'native': 'Français'},
            'de': {'name': 'German', 'native': 'Deutsch'},
            'hi': {'name': 'Hindi', 'native': 'हिन्दी'},
            'te': {'name': 'Telugu', 'native': 'తెలుగు'},  # For your location in TS
            'ta': {'name': 'Tamil', 'native': 'தமிழ்'},
            'zh': {'name': 'Chinese', 'native': '中文'},
            'ja': {'name': 'Japanese', 'native': '日本語'}
        }
        
        self.translations = self._load_translations()
        self.language_patterns = self._compile_language_patterns()
        
    def _load_translations(self) -> Dict:
        """Load comprehensive translation dictionaries"""
        
        translations = {
            'en': {
                'welcome': "Welcome to Internal Docs AI Assistant! I can help you with company policies, procedures, and documentation.",
                'processing': "🔄 Processing your request...",
                'no_results': "I don't have that information in the company documentation. Could you try rephrasing your question?",
                'error': "I encountered an error processing your request. Please try again or contact support.",
                'sources': "📚 Sources",
                'confidence': "🎯 Confidence",
                'response_time': "⏱️ Response time",
                'high_confidence': "High Confidence",
                'medium_confidence': "Medium Confidence", 
                'low_confidence': "Low Confidence",
                'help_message': "I can help with questions about company policies, HR procedures, technical documentation, and more. Just ask me anything!",
                'file_uploaded': "File uploaded successfully",
                'file_processing': "Processing uploaded file",
                'approval_needed': "This request requires approval",
                'approved': "Request approved",
                'rejected': "Request rejected"
            },
            
            'es': {
                'welcome': "¡Bienvenido al Asistente de Documentos Internos AI! Puedo ayudarte con políticas, procedimientos y documentación de la empresa.",
                'processing': "🔄 Procesando tu solicitud...",
                'no_results': "No tengo esa información en la documentación de la empresa. ¿Podrías reformular tu pregunta?",
                'error': "Encontré un error al procesar tu solicitud. Inténtalo de nuevo o contacta con soporte.",
                'sources': "📚 Fuentes",
                'confidence': "🎯 Confianza",
                'response_time': "⏱️ Tiempo de respuesta",
                'high_confidence': "Alta Confianza",
                'medium_confidence': "Confianza Media",
                'low_confidence': "Baja Confianza",
                'help_message': "Puedo ayudar con preguntas sobre políticas de la empresa, procedimientos de RRHH, documentación técnica y más. ¡Solo pregúntame!",
                'file_uploaded': "Archivo subido exitosamente",
                'file_processing': "Procesando archivo subido",
                'approval_needed': "Esta solicitud requiere aprobación",
                'approved': "Solicitud aprobada",
                'rejected': "Solicitud rechazada"
            },
            
            'hi': {
                'welcome': "आंतरिक दस्तावेज़ AI सहायक में आपका स्वागत है! मैं कंपनी की नीतियों, प्रक्रियाओं और दस्तावेज़ीकरण में आपकी सहायता कर सकता हूँ।",
                'processing': "🔄 आपका अनुरोध संसाधित कर रहा है...",
                'no_results': "मेरे पास कंपनी दस्तावेज़ीकरण में यह जानकारी नहीं है। क्या आप अपना प्रश्न दोबारा पूछ सकते हैं?",
                'error': "आपके अनुरोध को संसाधित करने में मुझे त्रुटि का सामना करना पड़ा। कृपया पुनः प्रयास करें।",
                'sources': "📚 स्रोत",
                'confidence': "🎯 विश्वास",
                'response_time': "⏱️ प्रतिक्रिया समय",
                'high_confidence': "उच्च विश्वास",
                'medium_confidence': "मध्यम विश्वास",
                'low_confidence': "कम विश्वास",
                'help_message': "मैं कंपनी नीतियों, HR प्रक्रियाओं, तकनीकी दस्तावेज़ीकरण और अधिक के बारे में प्रश्नों में सहायता कर सकता हूँ। बस मुझसे कुछ भी पूछें!",
                'file_uploaded': "फ़ाइल सफलतापूर्वक अपलोड हुई",
                'file_processing': "अपलोड की गई फ़ाइल का प्रसंस्करण",
                'approval_needed': "इस अनुरोध को अनुमोदन की आवश्यकता है",
                'approved': "अनुरोध स्वीकृत",
                'rejected': "अनुरोध अस्वीकृत"
            },
            
            'te': {  # Telugu for your location in Telangana
                'welcome': "అంతర్గత పత్రాలు AI సహాయకుడికి స్వాగతం! నేను కంపెనీ విధానాలు, ప్రక్రియలు మరియు డాక్యుమెంటేషన్‌లో మీకు సహాయం చేయగలను.",
                'processing': "🔄 మీ అభ్యర్థనను ప్రాసెస్ చేస్తున్నాను...",
                'no_results': "కంపెనీ డాక్యుమెంటేషన్‌లో నా దగ్గర ఆ సమాచారం లేదు. మీరు మీ ప్రశ్నను మళ్లీ అడగగలరా?",
                'error': "మీ అభ్యర్థనను ప్రాసెస్ చేయడంలో నాకు లోపం ఎదురైంది. దయచేసి మళ్లీ ప్రయత్నించండి.",
                'sources': "📚 మూలాలు",
                'confidence': "🎯 విశ్వాసం",
                'response_time': "⏱️ ప్రతిస్పందన సమయం",
                'high_confidence': "అధిక విశ్వాసం",
                'medium_confidence': "మధ్య విశ్వాసం",
                'low_confidence': "తక్కువ విశ్వాసం",
                'help_message': "నేను కంపెనీ విధానాలు, HR ప్రక్రియలు, సాంకేతిక డాక్యుమెంటేషన్ మరియు మరిన్ని గురించిన ప్రశ్నలలో సహాయం చేయగలను. నన్ను ఏదైనా అడగండి!",
                'file_uploaded': "ఫైల్ విజయవంతంగా అప్‌లోడ్ చేయబడింది",
                'file_processing': "అప్‌లోడ్ చేసిన ఫైల్‌ను ప్రాసెస్ చేస్తున్నాను",
                'approval_needed': "ఈ అభ్యర్థనకు అనుమతి అవసరం",
                'approved': "అభ్యర్థన ఆమోదించబడింది",
                'rejected': "అభ్యర్థన తిరస్కరించబడింది"
            },
            
            'fr': {
                'welcome': "Bienvenue dans l'Assistant IA de Documents Internes! Je peux vous aider avec les politiques, procédures et documentation de l'entreprise.",
                'processing': "🔄 Traitement de votre demande...",
                'no_results': "Je n'ai pas cette information dans la documentation de l'entreprise. Pourriez-vous reformuler votre question?",
                'error': "J'ai rencontré une erreur lors du traitement de votre demande. Veuillez réessayer.",
                'sources': "📚 Sources",
                'confidence': "🎯 Confiance",
                'response_time': "⏱️ Temps de réponse",
                'high_confidence': "Haute Confiance",
                'medium_confidence': "Confiance Moyenne",
                'low_confidence': "Faible Confiance",
                'help_message': "Je peux aider avec les questions sur les politiques d'entreprise, procédures RH, documentation technique et plus. Demandez-moi n'importe quoi!",
                'file_uploaded': "Fichier téléchargé avec succès",
                'file_processing': "Traitement du fichier téléchargé",
                'approval_needed': "Cette demande nécessite une approbation",
                'approved': "Demande approuvée",
                'rejected': "Demande rejetée"
            },
            
            'de': {
                'welcome': "Willkommen beim Internen Dokumenten-KI-Assistenten! Ich kann Ihnen bei Unternehmensrichtlinien, Verfahren und Dokumentation helfen.",
                'processing': "🔄 Verarbeitung Ihrer Anfrage...",
                'no_results': "Ich habe diese Information nicht in der Unternehmensdokumentation. Könnten Sie Ihre Frage umformulieren?",
                'error': "Bei der Verarbeitung Ihrer Anfrage ist ein Fehler aufgetreten. Bitte versuchen Sie es erneut.",
                'sources': "📚 Quellen",
                'confidence': "🎯 Vertrauen",
                'response_time': "⏱️ Antwortzeit",
                'high_confidence': "Hohes Vertrauen",
                'medium_confidence': "Mittleres Vertrauen",
                'low_confidence': "Geringes Vertrauen",
                'help_message': "Ich kann bei Fragen zu Unternehmensrichtlinien, HR-Verfahren, technischer Dokumentation und mehr helfen. Fragen Sie mich einfach!",
                'file_uploaded': "Datei erfolgreich hochgeladen",
                'file_processing': "Verarbeitung der hochgeladenen Datei",
                'approval_needed': "Diese Anfrage erfordert eine Genehmigung",
                'approved': "Anfrage genehmigt",
                'rejected': "Anfrage abgelehnt"
            }
        }
        
        return translations
    
    def _compile_language_patterns(self) -> Dict:
        """Compile regex patterns for language detection"""
        
        patterns = {
            'te': [
                r'[\u0C00-\u0C7F]+',  # Telugu script range
                r'\b(ఎలా|ఏమిటి|ఎందుకు|ఎప్పుడు|ఎక్కడ|ఎవరు|క్రింది|మీరు|నేను|మా|మీ)\b'
            ],
            'hi': [
                r'[\u0900-\u097F]+',  # Devanagari script range
                r'\b(कैसे|क्या|क्यों|कब|कहाँ|कौन|आप|मैं|हम|यह|वह)\b'
            ],
            'ta': [
                r'[\u0B80-\u0BFF]+',  # Tamil script range
                r'\b(எப்படி|என்ன|ஏன்|எப்போது|எங்கே|யார்|நீங்கள்|நான்|நாம்)\b'
            ],
            'es': [
                r'\b(cómo|qué|por qué|cuándo|dónde|quién|usted|tú|nosotros|política|empresa|beneficios)\b',
                r'\b(¿|¡|ñ|ó|é|í|á|ú)\b'
            ],
            'fr': [
                r'\b(comment|quoi|pourquoi|quand|où|qui|vous|nous|politique|entreprise|avantages)\b',
                r'\b(ç|é|è|ê|ë|î|ï|ô|ù|û|ü|ÿ|à|â)\b'
            ],
            'de': [
                r'\b(wie|was|warum|wann|wo|wer|Sie|wir|Politik|Unternehmen|Vorteile)\b',
                r'\b(ä|ö|ü|ß|Ä|Ö|Ü)\b'
            ]
        }
        
        # Compile patterns
        compiled_patterns = {}
        for lang, pattern_list in patterns.items():
            compiled_patterns[lang] = [re.compile(pattern, re.IGNORECASE) for pattern in pattern_list]
        
        return compiled_patterns
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """Detect language with confidence score"""
        
        if not text or len(text.strip()) < 3:
            return 'en', 0.5
        
        text = text.strip()
        scores = {'en': 0.1}  # Default English with low score
        
        # Check for script-based detection first (most reliable)
        for lang, patterns in self.language_patterns.items():
            score = 0
            text_length = len(text)
            
            for pattern in patterns:
                matches = pattern.findall(text)
                if matches:
                    # Calculate score based on match density
                    match_chars = sum(len(match) for match in matches)
                    score += (match_chars / text_length) * 10
            
            if score > 0:
                scores[lang] = min(score, 10)  # Cap at 10
        
        # Find language with highest score
        detected_lang = max(scores.items(), key=lambda x: x[1])
        
        # Normalize confidence to 0-1 range
        confidence = min(detected_lang[1] / 5, 1.0)
        
        return detected_lang[0], confidence
    
    def translate_message(self, message_key: str, target_language: str, **kwargs) -> str:
        """Get translated message for given key"""
        
        if target_language not in self.supported_languages:
            target_language = 'en'
        
        translations = self.translations.get(target_language, self.translations['en'])
        message = translations.get(message_key, message_key)
        
        # Format message with kwargs if provided
        try:
            if kwargs:
                message = message.format(**kwargs)
        except (KeyError, ValueError):
            pass  # Use original message if formatting fails
        
        return message
    
    def format_response_with_language(self, response_data: Dict, user_language: str) -> str:
        """Format RAG response with appropriate language"""
        
        answer = response_data.get('answer', '')
        sources = response_data.get('source_documents', [])
        confidence = response_data.get('confidence_score', 0.0)
        response_time = response_data.get('response_time', 0.0)
        
        # Translate system messages
        sources_label = self.translate_message('sources', user_language)
        confidence_label = self.translate_message('confidence', user_language)
        time_label = self.translate_message('response_time', user_language)
        
        # Determine confidence level translation
        if confidence > 0.7:
            confidence_text = self.translate_message('high_confidence', user_language)
            confidence_emoji = "🟢"
        elif confidence > 0.4:
            confidence_text = self.translate_message('medium_confidence', user_language)
            confidence_emoji = "🟡"
        else:
            confidence_text = self.translate_message('low_confidence', user_language)
            confidence_emoji = "🔴"
        
        # Build formatted response
        formatted_response = f"🤖 {answer}\n\n"
        formatted_response += f"{confidence_emoji} {confidence_text} | {time_label}: {response_time:.1f}s | {sources_label}: {len(sources)}\n"
        
        # Add sources if available
        if sources:
            formatted_response += f"\n📚 {sources_label}:\n"
            for i, doc in enumerate(sources[:3]):
                source_file = doc.metadata.get('source_file', 'Unknown')
                preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                formatted_response += f"• `{source_file}`: {preview}\n"
        
        return formatted_response
    
    def get_language_selection_blocks(self, current_language: str = 'en') -> List[Dict]:
        """Generate Slack blocks for language selection"""
        
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "🌐 Select Your Language / भाषा चुनें / భాష ఎంచుకోండి"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"Current language: *{self.supported_languages[current_language]['native']}*\n\nSelect your preferred language for interactions:"
                }
            }
        ]
        
        # Create language selection buttons
        elements = []
        for lang_code, lang_info in self.supported_languages.items():
            elements.append({
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": f"{lang_info['native']}"
                },
                "value": lang_code,
                "action_id": f"select_language_{lang_code}"
            })
            
            # Add new row after every 3 buttons
            if len(elements) == 3:
                blocks.append({
                    "type": "actions",
                    "elements": elements.copy()
                })
                elements.clear()
        
        # Add remaining buttons
        if elements:
            blocks.append({
                "type": "actions",
                "elements": elements
            })
        
        return blocks
    
    def get_supported_languages_info(self) -> str:
        """Get information about supported languages"""
        
        info = "🌐 **Supported Languages:**\n\n"
        
        for lang_code, lang_info in self.supported_languages.items():
            info += f"• **{lang_info['name']}** ({lang_info['native']}) - Code: `{lang_code}`\n"
        
        info += "\n**Language Detection:**\n"
        info += "• Automatic detection based on text content\n"
        info += "• Script-based detection for Indian languages\n"
        info += "• Keyword pattern matching for European languages\n"
        info += "• Confidence scoring for accuracy\n"
        
        info += "\n**Features:**\n"
        info += "• System messages translated to your language\n"
        info += "• UI elements localized\n"
        info += "• Help content in native language\n"
        info += "• Error messages translated\n"
        
        return info
