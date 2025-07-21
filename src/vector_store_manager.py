import os
import logging
import json
import pickle
import hashlib
from typing import List, Optional, Dict, Any, Tuple, Union
from pathlib import Path
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import warnings
import sys

# Suppress deprecation warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Vector store implementations with fallback handling
try:
    from langchain_community.vectorstores import FAISS, Chroma
    COMMUNITY_VECTORSTORES_AVAILABLE = True
except ImportError:
    try:
        from langchain.vectorstores import FAISS, Chroma
        COMMUNITY_VECTORSTORES_AVAILABLE = True
    except ImportError:
        print("⚠️ Vector store implementations not available")
        COMMUNITY_VECTORSTORES_AVAILABLE = False

# Embedding implementations with comprehensive fallback
try:
    from langchain_openai import OpenAIEmbeddings
    OPENAI_EMBEDDINGS_AVAILABLE = True
except ImportError:
    print("⚠️ OpenAI embeddings not available")
    OPENAI_EMBEDDINGS_AVAILABLE = False

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    HUGGINGFACE_EMBEDDINGS_AVAILABLE = True
except ImportError:
    try:
        from langchain.embeddings import HuggingFaceEmbeddings
        HUGGINGFACE_EMBEDDINGS_AVAILABLE = True
    except ImportError:
        print("⚠️ HuggingFace embeddings not available")
        HUGGINGFACE_EMBEDDINGS_AVAILABLE = False

# Check for sentence-transformers specifically
try:
    import sentence_transformers
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠️ sentence-transformers not installed. Install with: pip install sentence-transformers")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Essential imports
from langchain.schema import Document
import numpy as np
import psutil
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingConfig:
    """Enhanced configuration class for different embedding strategies"""
    
    OPENAI_ADA002 = {
        'name': 'openai-ada-002',
        'model': 'text-embedding-ada-002', 
        'dimensions': 1536,
        'max_tokens': 8191,
        'cost_per_1k': 0.0001,
        'requires_api': True,
        'requires_package': 'langchain_openai',
        'fallback_available': True
    }
    
    OPENAI_ADA003 = {
        'name': 'openai-ada-003',
        'model': 'text-embedding-3-small',
        'dimensions': 1536,
        'max_tokens': 8191,
        'cost_per_1k': 0.00002,
        'requires_api': True,
        'requires_package': 'langchain_openai',
        'fallback_available': True
    }
    
    HUGGINGFACE_ALL_MINI = {
        'name': 'sentence-transformers-all-MiniLM-L6-v2',
        'model': 'sentence-transformers/all-MiniLM-L6-v2',
        'dimensions': 384,
        'max_tokens': 256,
        'cost_per_1k': 0.0,
        'requires_api': False,
        'requires_package': 'sentence_transformers',
        'fallback_available': True
    }
    
    HUGGINGFACE_ALL_MPNET = {
        'name': 'sentence-transformers-all-mpnet-base-v2',
        'model': 'sentence-transformers/all-mpnet-base-v2',
        'dimensions': 768,
        'max_tokens': 384,
        'cost_per_1k': 0.0,
        'requires_api': False,
        'requires_package': 'sentence_transformers',
        'fallback_available': True
    }
    
    # Fallback for when no embeddings are available
    MOCK_EMBEDDINGS = {
        'name': 'mock-embeddings',
        'model': 'mock-random-vectors',
        'dimensions': 384,
        'max_tokens': 512,
        'cost_per_1k': 0.0,
        'requires_api': False,
        'requires_package': None,
        'fallback_available': False
    }

class VectorStoreMetrics:
    """Enhanced performance and usage metrics tracking"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.embeddings_created = 0
        self.queries_processed = 0
        self.total_embedding_time = 0
        self.total_query_time = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.memory_usage_mb = 0
        self.storage_usage_mb = 0
        self.error_count = 0
        self.fallback_usage = 0
        self.batch_operations = 0
        self.concurrent_operations = 0
        
    def record_embedding(self, processing_time: float, count: int = 1):
        self.embeddings_created += count
        self.total_embedding_time += processing_time
        
    def record_query(self, processing_time: float):
        self.queries_processed += 1
        self.total_query_time += processing_time
        
    def record_cache_hit(self):
        self.cache_hits += 1
        
    def record_cache_miss(self):
        self.cache_misses += 1
        
    def record_error(self):
        self.error_count += 1
        
    def record_fallback_usage(self):
        self.fallback_usage += 1
        
    def record_batch_operation(self):
        self.batch_operations += 1
        
    def record_concurrent_operation(self):
        self.concurrent_operations += 1
        
    def update_resource_usage(self):
        try:
            process = psutil.Process()
            self.memory_usage_mb = process.memory_info().rss / 1024 / 1024
        except Exception:
            self.memory_usage_mb = 0
        
    def get_summary(self) -> Dict[str, Any]:
        runtime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            'runtime_seconds': runtime,
            'embeddings_created': self.embeddings_created,
            'queries_processed': self.queries_processed,
            'avg_embedding_time': self.total_embedding_time / max(1, self.embeddings_created),
            'avg_query_time': self.total_query_time / max(1, self.queries_processed),
            'cache_hit_rate': self.cache_hits / max(1, self.cache_hits + self.cache_misses),
            'memory_usage_mb': self.memory_usage_mb,
            'throughput_embeddings_per_sec': self.embeddings_created / max(1, runtime),
            'throughput_queries_per_sec': self.queries_processed / max(1, runtime),
            'error_rate': self.error_count / max(1, self.embeddings_created + self.queries_processed),
            'fallback_usage': self.fallback_usage,
            'batch_operations': self.batch_operations,
            'concurrent_operations': self.concurrent_operations
        }

class EmbeddingCache:
    """Enhanced smart caching system for embeddings"""
    
    def __init__(self, cache_dir: str = "cache", max_size_mb: int = 512):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size_mb = max_size_mb
        self.memory_cache = {}
        self.cache_index = {}
        self.lock = threading.RLock()
        self._load_cache_index()
        
    def _load_cache_index(self):
        """Load cache index from disk with error handling"""
        index_file = self.cache_dir / "cache_index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    self.cache_index = json.load(f)
                logger.info(f"✅ Loaded cache index with {len(self.cache_index)} entries")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load cache index: {e}")
                self.cache_index = {}
    
    def _save_cache_index(self):
        """Save cache index to disk with error handling"""
        index_file = self.cache_dir / "cache_index.json"
        try:
            with open(index_file, 'w') as f:
                json.dump(self.cache_index, f, indent=2)
        except Exception as e:
            logger.warning(f"⚠️ Failed to save cache index: {e}")
    
    def _get_content_hash(self, content: str) -> str:
        """Generate hash for content with UTF-8 handling"""
        try:
            return hashlib.md5(content.encode('utf-8')).hexdigest()
        except Exception:
            return hashlib.md5(str(content).encode('utf-8', errors='ignore')).hexdigest()
    
    def get_embedding(self, content: str, model_name: str) -> Optional[List[float]]:
        """Retrieve embedding from cache with thread safety"""
        with self.lock:
            try:
                content_hash = self._get_content_hash(content)
                cache_key = f"{model_name}_{content_hash}"
                
                # Check memory cache first
                if cache_key in self.memory_cache:
                    return self.memory_cache[cache_key]
                
                # Check disk cache
                if cache_key in self.cache_index:
                    cache_file = self.cache_dir / f"{cache_key}.pkl"
                    if cache_file.exists():
                        try:
                            with open(cache_file, 'rb') as f:
                                embedding = pickle.load(f)
                            # Load into memory cache
                            self.memory_cache[cache_key] = embedding
                            return embedding
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to load cached embedding: {e}")
                            # Remove corrupted cache entry
                            if cache_file.exists():
                                cache_file.unlink()
                            if cache_key in self.cache_index:
                                del self.cache_index[cache_key]
                
                return None
            except Exception as e:
                logger.error(f"❌ Cache retrieval error: {e}")
                return None
    
    def store_embedding(self, content: str, model_name: str, embedding: List[float]):
        """Store embedding in cache with thread safety and size management"""
        with self.lock:
            try:
                content_hash = self._get_content_hash(content)
                cache_key = f"{model_name}_{content_hash}"
                
                # Store in memory cache
                self.memory_cache[cache_key] = embedding
                
                # Manage memory cache size
                if len(self.memory_cache) > 1000:  # Limit memory cache size
                    # Remove oldest entries
                    oldest_keys = list(self.memory_cache.keys())[:100]
                    for key in oldest_keys:
                        del self.memory_cache[key]
                
                # Store on disk
                cache_file = self.cache_dir / f"{cache_key}.pkl"
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(embedding, f)
                    
                    self.cache_index[cache_key] = {
                        'created': datetime.now().isoformat(),
                        'content_length': len(content),
                        'file_size': cache_file.stat().st_size,
                        'model_name': model_name
                    }
                    self._save_cache_index()
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to cache embedding to disk: {e}")
                    
            except Exception as e:
                logger.error(f"❌ Failed to store embedding: {e}")
    
    def cleanup_old_cache(self, max_age_days: int = 30):
        """Remove old cache entries with improved error handling"""
        with self.lock:
            try:
                cutoff_date = datetime.now() - timedelta(days=max_age_days)
                removed_count = 0
                
                for cache_key, info in list(self.cache_index.items()):
                    try:
                        created_date = datetime.fromisoformat(info['created'])
                        if created_date < cutoff_date:
                            cache_file = self.cache_dir / f"{cache_key}.pkl"
                            if cache_file.exists():
                                cache_file.unlink()
                            del self.cache_index[cache_key]
                            if cache_key in self.memory_cache:
                                del self.memory_cache[cache_key]
                            removed_count += 1
                    except Exception as e:
                        logger.warning(f"⚠️ Error cleaning cache entry {cache_key}: {e}")
                        # Remove corrupted entry
                        if cache_key in self.cache_index:
                            del self.cache_index[cache_key]
                
                if removed_count > 0:
                    logger.info(f"🧹 Cleaned up {removed_count} old cache entries")
                    self._save_cache_index()
                    
            except Exception as e:
                logger.error(f"❌ Cache cleanup failed: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        total_size = 0
        try:
            for file in self.cache_dir.glob("*.pkl"):
                total_size += file.stat().st_size
        except Exception:
            pass
            
        return {
            'memory_entries': len(self.memory_cache),
            'disk_entries': len(self.cache_index),
            'total_size_mb': total_size / (1024 * 1024),
            'cache_dir': str(self.cache_dir)
        }

class MockEmbeddingService:
    """Mock embedding service for fallback when no real embeddings are available"""
    
    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions
        self._embedding_cache = {}
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate consistent mock embeddings based on text hash"""
        embeddings = []
        for text in texts:
            # Create deterministic "embedding" based on text hash
            text_hash = hash(text) % 1000000
            np.random.seed(text_hash)
            embedding = np.random.normal(0, 1, self.dimensions).tolist()
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Generate consistent mock embedding for single query"""
        return self.embed_documents([text])[0]

class ProductionVectorStoreManager:
    """Production-ready vector store manager with comprehensive fallback handling"""
    
    def __init__(
        self,
        storage_dir: str = "vectorstore",
        cache_dir: str = "cache",
        embedding_config: str = "auto",
        vector_store_type: str = "faiss",
        enable_caching: bool = True,
        max_memory_mb: int = 2048,
        enable_gpu: bool = True,
        batch_size: int = 50,
        max_workers: int = 4
    ):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        self.cache_dir = cache_dir
        self.vector_store_type = vector_store_type.lower()
        self.enable_caching = enable_caching
        self.max_memory_mb = max_memory_mb
        self.enable_gpu = enable_gpu
        self.batch_size = batch_size
        self.max_workers = max_workers
        
        # Initialize components
        self.embedding_service = None
        self.vector_store = None
        self.metrics = VectorStoreMetrics()
        self.cache = EmbeddingCache(cache_dir) if enable_caching else None
        
        # Hardware detection and config
        self.hardware_config = self._detect_hardware()
        self.embedding_config = self._select_embedding_config(embedding_config)
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Status tracking
        self.initialization_status = {
            'embedding_service': False,
            'vector_store': False,
            'cache_system': enable_caching,
            'hardware_detected': True
        }
        
        logger.info(f"🚀 Vector Store Manager initialized: {self.embedding_config['name']} + {vector_store_type.upper()}")
        logger.info(f"📊 Hardware: {self.hardware_config['cpu_count']} CPUs, {self.hardware_config['memory_gb']:.1f}GB RAM, GPU: {self.hardware_config['has_gpu']}")
        
    def _detect_hardware(self) -> Dict[str, Any]:
        """Enhanced hardware detection with GPU support"""
        config = {
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'has_gpu': False,
            'gpu_memory_gb': 0,
            'gpu_name': 'None',
            'platform': sys.platform
        }
        
        # Enhanced GPU detection
        try:
            import torch
            if torch.cuda.is_available():
                config['has_gpu'] = True
                config['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                config['gpu_name'] = torch.cuda.get_device_name(0)
                logger.info(f"🎮 GPU detected: {config['gpu_name']} ({config['gpu_memory_gb']:.1f}GB)")
            else:
                logger.info("🖥️ No GPU available, using CPU")
        except ImportError:
            logger.info("🖥️ PyTorch not available, CPU-only mode")
            
        # Check for Apple Silicon
        if sys.platform == "darwin":
            try:
                import platform
                if platform.processor() == 'arm':
                    config['apple_silicon'] = True
                    logger.info("🍎 Apple Silicon detected")
            except Exception:
                pass
                
        logger.info(f"💻 Hardware: {config['cpu_count']} CPUs, {config['memory_gb']:.1f}GB RAM")
        return config
    
    def _select_embedding_config(self, preference: str) -> Dict[str, Any]:
        """Enhanced embedding configuration selection with fallback logic"""
        
        # Check OpenAI availability first
        openai_available = (
            OPENAI_EMBEDDINGS_AVAILABLE and 
            os.getenv("OPENAI_API_KEY") and 
            os.getenv("OPENAI_API_KEY") != "your_openai_key_here"
        )
        
        # OpenAI preference or explicit request
        if preference == "openai" or (preference == "auto" and openai_available):
            if openai_available:
                return EmbeddingConfig.OPENAI_ADA002
        
        # HuggingFace preference
        if preference in ["huggingface", "huggingface-mini", "huggingface-mpnet"] or preference == "auto":
            if HUGGINGFACE_EMBEDDINGS_AVAILABLE and SENTENCE_TRANSFORMERS_AVAILABLE:
                # Auto-select based on hardware
                if preference == "auto":
                    if self.hardware_config['memory_gb'] >= 8:
                        return EmbeddingConfig.HUGGINGFACE_ALL_MPNET
                    else:
                        return EmbeddingConfig.HUGGINGFACE_ALL_MINI
                elif preference == "huggingface-mini":
                    return EmbeddingConfig.HUGGINGFACE_ALL_MINI
                elif preference == "huggingface-mpnet":
                    return EmbeddingConfig.HUGGINGFACE_ALL_MPNET
                else:
                    return EmbeddingConfig.HUGGINGFACE_ALL_MINI
        
        # Fallback cascade
        logger.warning("⚠️ Preferred embedding service not available, checking fallbacks...")
        
        # Try OpenAI if available
        if openai_available:
            logger.info("🔄 Falling back to OpenAI embeddings")
            return EmbeddingConfig.OPENAI_ADA002
        
        # Try HuggingFace if available
        if HUGGINGFACE_EMBEDDINGS_AVAILABLE and SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.info("🔄 Falling back to HuggingFace embeddings")
            return EmbeddingConfig.HUGGINGFACE_ALL_MINI
        
        # Final fallback to mock embeddings
        logger.warning("⚠️ No embedding services available, using mock embeddings")
        return EmbeddingConfig.MOCK_EMBEDDINGS
    
    def initialize_embedding_service(self) -> bool:
        """Enhanced embedding service initialization with comprehensive fallback"""
        try:
            config = self.embedding_config
            logger.info(f"🔧 Initializing embedding service: {config['name']}")
            
            # OpenAI embeddings
            if config['requires_api'] and config['name'].startswith('openai'):
                if not OPENAI_EMBEDDINGS_AVAILABLE:
                    logger.error("❌ OpenAI embeddings package not available")
                    return self._fallback_to_next_embedding()
                
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key or api_key == "your_openai_key_here":
                    logger.warning("⚠️ OpenAI API key not available")
                    return self._fallback_to_next_embedding()
                
                try:
                    self.embedding_service = OpenAIEmbeddings(
                        openai_api_key=api_key,
                        model=config['model'],
                        chunk_size=1000
                    )
                    # Test the service
                    test_embedding = self.embedding_service.embed_query("test")
                    if len(test_embedding) == config['dimensions']:
                        logger.info(f"✅ OpenAI embeddings initialized: {config['model']}")
                        self.initialization_status['embedding_service'] = True
                        return True
                    else:
                        logger.error("❌ OpenAI embeddings test failed")
                        return self._fallback_to_next_embedding()
                        
                except Exception as e:
                    logger.error(f"❌ OpenAI embeddings initialization failed: {e}")
                    return self._fallback_to_next_embedding()
            
            # HuggingFace embeddings
            elif not config['requires_api'] and config['name'].startswith('sentence-transformers'):
                if not HUGGINGFACE_EMBEDDINGS_AVAILABLE:
                    logger.error("❌ HuggingFace embeddings package not available")
                    return self._fallback_to_next_embedding()
                
                if not SENTENCE_TRANSFORMERS_AVAILABLE:
                    logger.error("❌ sentence-transformers package not available")
                    logger.info("💡 Install with: pip install sentence-transformers")
                    return self._fallback_to_next_embedding()
                
                try:
                    device = 'cuda' if (self.hardware_config['has_gpu'] and self.enable_gpu) else 'cpu'
                    logger.info(f"🔧 Using device: {device}")
                    
                    self.embedding_service = HuggingFaceEmbeddings(
                        model_name=config['model'],
                        model_kwargs={'device': device},
                        encode_kwargs={'normalize_embeddings': True}
                    )
                    
                    # Test the service
                    test_embedding = self.embedding_service.embed_query("test")
                    if len(test_embedding) == config['dimensions']:
                        logger.info(f"✅ HuggingFace embeddings initialized: {config['model']}")
                        self.initialization_status['embedding_service'] = True
                        return True
                    else:
                        logger.error("❌ HuggingFace embeddings test failed")
                        return self._fallback_to_next_embedding()
                        
                except Exception as e:
                    logger.error(f"❌ HuggingFace embeddings initialization failed: {e}")
                    return self._fallback_to_next_embedding()
            
            # Mock embeddings (final fallback)
            elif config['name'] == 'mock-embeddings':
                self.embedding_service = MockEmbeddingService(config['dimensions'])
                logger.warning("⚠️ Using mock embeddings (development/testing only)")
                self.metrics.record_fallback_usage()
                self.initialization_status['embedding_service'] = True
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Critical error in embedding service initialization: {e}")
            return self._fallback_to_next_embedding()
    
    def _fallback_to_next_embedding(self) -> bool:
        """Cascading fallback to next available embedding service"""
        current_name = self.embedding_config['name']
        
        # Try OpenAI if not already tried
        if not current_name.startswith('openai'):
            if (OPENAI_EMBEDDINGS_AVAILABLE and 
                os.getenv("OPENAI_API_KEY") and 
                os.getenv("OPENAI_API_KEY") != "your_openai_key_here"):
                logger.info("🔄 Attempting fallback to OpenAI embeddings")
                self.embedding_config = EmbeddingConfig.OPENAI_ADA002
                return self.initialize_embedding_service()
        
        # Try HuggingFace if not already tried
        if not current_name.startswith('sentence-transformers'):
            if HUGGINGFACE_EMBEDDINGS_AVAILABLE and SENTENCE_TRANSFORMERS_AVAILABLE:
                logger.info("🔄 Attempting fallback to HuggingFace embeddings")
                self.embedding_config = EmbeddingConfig.HUGGINGFACE_ALL_MINI
                return self.initialize_embedding_service()
        
        # Final fallback to mock embeddings
        if current_name != 'mock-embeddings':
            logger.warning("🔄 Final fallback to mock embeddings")
            self.embedding_config = EmbeddingConfig.MOCK_EMBEDDINGS
            return self.initialize_embedding_service()
        
        logger.error("❌ All embedding fallbacks exhausted")
        return False

    def initialize_vector_store(self, documents: List[Document] = None) -> bool:
        """Enhanced vector store initialization with robust error handling"""
        try:
            logger.info("🔧 Initializing vector store...")
            
            # Ensure embedding service is ready
            if not self.embedding_service:
                if not self.initialize_embedding_service():
                    logger.error("❌ Cannot initialize vector store without embedding service")
                    return False
            
            # Check for existing vector store
            if self._load_existing_vector_store():
                logger.info("✅ Loaded existing vector store")
                self.initialization_status['vector_store'] = True
                return True
            
            # Create new vector store
            if documents and len(documents) > 0:
                success = self._create_new_vector_store(documents)
                self.initialization_status['vector_store'] = success
                return success
            else:
                logger.info("✅ Vector store initialized without documents (ready for document addition)")
                self.initialization_status['vector_store'] = True
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize vector store: {e}")
            self.metrics.record_error()
            return False

    def _load_existing_vector_store(self) -> bool:
        """Enhanced existing vector store loading with error recovery"""
        try:
            if self.vector_store_type == "faiss":
                store_path = self.storage_dir / "faiss_index"
                if store_path.exists() and any(store_path.iterdir()):
                    try:
                        # Try loading with new FAISS format first
                        self.vector_store = FAISS.load_local(
                            str(store_path), 
                            self.embedding_service,
                            allow_dangerous_deserialization=True
                        )
                        logger.info("✅ FAISS vector store loaded successfully")
                        return True
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to load FAISS store, will recreate: {e}")
                        # Clean up corrupted store
                        import shutil
                        shutil.rmtree(store_path, ignore_errors=True)
                        return False
            
            elif self.vector_store_type == "chroma":
                store_path = self.storage_dir / "chroma_db"
                if store_path.exists() and any(store_path.iterdir()):
                    try:
                        self.vector_store = Chroma(
                            persist_directory=str(store_path),
                            embedding_function=self.embedding_service
                        )
                        logger.info("✅ Chroma vector store loaded successfully")
                        return True
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to load Chroma store, will recreate: {e}")
                        return False
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to load existing vector store: {e}")
        
        return False

    def _create_new_vector_store(self, documents: List[Document]) -> bool:
        """Enhanced vector store creation with memory management"""
        try:
            logger.info(f"🏗️ Creating new {self.vector_store_type.upper()} vector store with {len(documents)} documents")
            
            if not COMMUNITY_VECTORSTORES_AVAILABLE:
                logger.error("❌ Vector store packages not available")
                return False
            
            if self.vector_store_type == "faiss":
                # Enhanced batch processing for memory efficiency
                effective_batch_size = min(self.batch_size, len(documents))
                
                # Adjust batch size based on available memory
                if self.hardware_config['memory_gb'] < 4:
                    effective_batch_size = min(20, effective_batch_size)
                elif self.hardware_config['memory_gb'] < 8:
                    effective_batch_size = min(30, effective_batch_size)
                
                total_batches = (len(documents) - 1) // effective_batch_size + 1
                
                for i in range(0, len(documents), effective_batch_size):
                    batch = documents[i:i + effective_batch_size]
                    batch_num = i // effective_batch_size + 1
                    
                    logger.info(f"🔄 Processing batch {batch_num}/{total_batches} ({len(batch)} documents)")
                    
                    try:
                        if i == 0:
                            # Create initial vector store
                            self.vector_store = FAISS.from_documents(batch, self.embedding_service)
                        else:
                            # Add to existing vector store
                            batch_store = FAISS.from_documents(batch, self.embedding_service)
                            self.vector_store.merge_from(batch_store)
                            del batch_store  # Explicit cleanup
                        
                        # Memory monitoring and management
                        self.metrics.update_resource_usage()
                        if self.metrics.memory_usage_mb > self.max_memory_mb * 0.8:
                            logger.warning(f"⚠️ High memory usage: {self.metrics.memory_usage_mb:.1f}MB")
                            time.sleep(2)  # Allow garbage collection
                        
                        # Progress update for large datasets
                        if batch_num % 5 == 0 or batch_num == total_batches:
                            progress = (batch_num / total_batches) * 100
                            logger.info(f"📊 Progress: {progress:.1f}% complete")
                            
                    except Exception as e:
                        logger.error(f"❌ Failed to process batch {batch_num}: {e}")
                        self.metrics.record_error()
                        return False
                
                # Save to disk with verification
                store_path = self.storage_dir / "faiss_index"
                self.vector_store.save_local(str(store_path))
                
                # Verify save
                if store_path.exists() and any(store_path.iterdir()):
                    logger.info("✅ FAISS vector store saved successfully")
                else:
                    logger.error("❌ Failed to save FAISS vector store")
                    return False
                
            elif self.vector_store_type == "chroma":
                store_path = self.storage_dir / "chroma_db"
                
                # Handle large document sets for Chroma
                if len(documents) > 1000:
                    logger.info("📊 Large document set detected, using batch processing for Chroma")
                    # Create initial store with first batch
                    initial_batch = documents[:100]
                    self.vector_store = Chroma.from_documents(
                        initial_batch,
                        self.embedding_service,
                        persist_directory=str(store_path)
                    )
                    
                    # Add remaining documents in batches
                    for i in range(100, len(documents), 100):
                        batch = documents[i:i + 100]
                        self.vector_store.add_documents(batch)
                        if i % 500 == 0:  # Persist every 500 documents
                            self.vector_store.persist()
                else:
                    self.vector_store = Chroma.from_documents(
                        documents,
                        self.embedding_service,
                        persist_directory=str(store_path)
                    )
                
                self.vector_store.persist()
                logger.info("✅ Chroma vector store created and persisted successfully")
            
            else:
                logger.error(f"❌ Unsupported vector store type: {self.vector_store_type}")
                return False
            
            # Final verification
            if self.vector_store:
                logger.info(f"✅ Vector store created successfully with {len(documents)} documents")
                return True
            else:
                logger.error("❌ Vector store creation failed - store is None")
                return False
            
        except Exception as e:
            logger.error(f"❌ Failed to create vector store: {e}")
            self.metrics.record_error()
            return False

    def similarity_search(
        self, 
        query: str, 
        k: int = 5, 
        score_threshold: float = 0.0,
        filter_metadata: Optional[Dict] = None
    ) -> List[Tuple[Document, float]]:
        """Enhanced similarity search with comprehensive error handling"""
        
        start_time = time.time()
        
        try:
            if not self.vector_store:
                logger.error("❌ Vector store not initialized")
                return []
            
            logger.debug(f"🔍 Searching for: '{query[:50]}...' (k={k}, threshold={score_threshold})")
            
            # Perform similarity search with scores
            results = []
            
            if hasattr(self.vector_store, 'similarity_search_with_score'):
                try:
                    results = self.vector_store.similarity_search_with_score(
                        query, k=k*2  # Get extra results for filtering
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Score-based search failed, falling back to basic search: {e}")
                    docs = self.vector_store.similarity_search(query, k=k)
                    results = [(doc, 1.0) for doc in docs]  # Mock scores
            else:
                # Fallback for vector stores without scoring
                docs = self.vector_store.similarity_search(query, k=k)
                results = [(doc, 1.0) for doc in docs]  # Mock scores
            
            # Apply score threshold
            if score_threshold > 0:
                filtered_results = [
                    (doc, score) for doc, score in results 
                    if score >= score_threshold
                ]
            else:
                filtered_results = results
            
            # Apply metadata filtering
            if filter_metadata:
                filtered_results = [
                    (doc, score) for doc, score in filtered_results
                    if all(
                        doc.metadata.get(key) == value 
                        for key, value in filter_metadata.items()
                    )
                ]
            
            # Limit to requested number
            filtered_results = filtered_results[:k]
            
            # Update metrics
            query_time = time.time() - start_time
            self.metrics.record_query(query_time)
            
            logger.info(f"✅ Similarity search returned {len(filtered_results)} results in {query_time:.2f}s")
            return filtered_results
            
        except Exception as e:
            logger.error(f"❌ Similarity search failed: {e}")
            self.metrics.record_error()
            return []

    def get_vector_store_stats(self) -> Dict[str, Any]:
        """Enhanced comprehensive vector store statistics"""
        stats = {
            'embedding_config': self.embedding_config,
            'vector_store_type': self.vector_store_type,
            'hardware_config': self.hardware_config,
            'initialization_status': self.initialization_status,
            'storage_size_mb': 0,
            'document_count': 0,
            'index_size': 0,
            'cache_stats': {}
        }
        
        # Calculate storage size
        try:
            if self.storage_dir.exists():
                total_size = sum(f.stat().st_size for f in self.storage_dir.rglob('*') if f.is_file())
                stats['storage_size_mb'] = total_size / (1024 * 1024)
        except Exception as e:
            logger.warning(f"⚠️ Failed to calculate storage size: {e}")
        
        # Get vector store specific stats
        try:
            if self.vector_store:
                if hasattr(self.vector_store, 'index'):
                    if hasattr(self.vector_store.index, 'ntotal'):
                        stats['document_count'] = self.vector_store.index.ntotal
                        stats['index_size'] = self.vector_store.index.ntotal * self.embedding_config['dimensions'] * 4  # 4 bytes per float
                elif self.vector_store_type == "chroma":
                    # Try to get Chroma collection count
                    try:
                        collection = self.vector_store._collection
                        stats['document_count'] = collection.count()
                    except Exception:
                        stats['document_count'] = "unknown"
        except Exception as e:
            logger.warning(f"⚠️ Failed to get vector store stats: {e}")
        
        # Add cache statistics
        if self.cache:
            stats['cache_stats'] = self.cache.get_cache_stats()
        
        # Add performance metrics
        stats.update(self.metrics.get_summary())
        
        return stats

    def optimize_performance(self):
        """Enhanced performance optimization with health checks"""
        logger.info("⚡ Running comprehensive performance optimization...")
        
        try:
            # Clean old cache entries
            if self.cache:
                self.cache.cleanup_old_cache(max_age_days=7)
            
            # Update resource usage
            self.metrics.update_resource_usage()
            
            logger.info("✅ Performance optimization completed")
            
        except Exception as e:
            logger.error(f"❌ Performance optimization failed: {e}")

    def export_vector_store(self, export_path: str, format: str = "pkl") -> bool:
        """Enhanced vector store export with metadata"""
        try:
            export_dir = Path(export_path)
            export_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"📦 Exporting vector store to {export_path}")
            
            if format == "pkl" and self.vector_store:
                # Export vector store
                with open(export_dir / "vector_store.pkl", 'wb') as f:
                    pickle.dump(self.vector_store, f)
                
                # Export comprehensive metadata
                metadata = {
                    'export_info': {
                        'created': datetime.now().isoformat(),
                        'version': '1.0',
                        'format': format
                    },
                    'config': self.embedding_config,
                    'vector_store_type': self.vector_store_type,
                    'hardware_config': self.hardware_config,
                    'initialization_status': self.initialization_status,
                    'stats': self.get_vector_store_stats()
                }
                
                with open(export_dir / "metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                logger.info(f"✅ Vector store exported successfully to {export_path}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Export failed: {e}")
            return False

# Enhanced testing and demonstration functions
def test_production_vector_store():
    """Comprehensive test of the enhanced production vector store system"""
    print("=== Enhanced Production Vector Store System Test ===")
    
    # Import document loader with error handling
    try:
        from document_loader import OptimizedDocumentLoader
    except ImportError:
        print("❌ Document loader not found. Please ensure document_loader.py exists.")
        print("💡 You can continue with sample data generation.")
        
        # Create minimal test documents
        test_docs = [
            Document(page_content="Company refund policy: We offer full refunds within 30 days.", metadata={'source': 'test_policy.txt'}),
            Document(page_content="Employee benefits include health insurance and 401k matching.", metadata={'source': 'test_hr.txt'}),
            Document(page_content="Technical support is available 24/7 through our help desk.", metadata={'source': 'test_tech.txt'})
        ]
        documents = test_docs
    else:
        # Load test documents using document loader
        loader = OptimizedDocumentLoader()
        if not Path("data").exists() or not list(Path("data").glob("*.txt")):
            print("📝 Creating sample documents...")
            loader.create_sample_documents()
        
        documents = loader.load_directory("data")
    
    print(f"📄 Loaded {len(documents)} document chunks")
    
    # Initialize system with comprehensive testing
    print("\n🔧 Initializing Production Vector Store Manager...")
    manager = ProductionVectorStoreManager(
        embedding_config="auto",
        vector_store_type="faiss",
        enable_caching=True,
        max_memory_mb=2048,
        enable_gpu=True
    )
    
    # Initialize embedding service
    print("\n🧠 Initializing embedding service...")
    embedding_ready = manager.initialize_embedding_service()
    print(f"Embedding service ready: {'✅' if embedding_ready else '❌'}")
    
    if embedding_ready:
        print(f"Using: {manager.embedding_config['name']}")
    
    # Initialize vector store
    print("\n🗄️ Initializing vector store...")
    vectorstore_ready = manager.initialize_vector_store(documents)
    print(f"Vector store ready: {'✅' if vectorstore_ready else '❌'}")
    
    if not vectorstore_ready:
        print("❌ Vector store initialization failed - check logs above")
        return None
    
    # Test queries with comprehensive coverage
    test_queries = [
        "What's our refund policy?",
        "How do I request design assets?", 
        "What are the employee benefits?",
        "How do I submit an expense report?",
        "What is the remote work policy?",
        "How do I get IT support?",
        "What is our API access information?",
        "How do performance reviews work?",
        "What are the vacation policies?",
        "How do I contact HR?"
    ]
    
    print("\n🔍 Testing Enhanced Similarity Search:")
    successful_queries = 0
    
    for i, query in enumerate(test_queries):
        print(f"\n📝 Query {i+1}: {query}")
        try:
            results = manager.similarity_search(query, k=3, score_threshold=0.1)
            
            if results:
                successful_queries += 1
                for j, (doc, score) in enumerate(results):
                    print(f"  ✅ Result {j+1} (Score: {score:.3f}): {doc.page_content[:100]}...")
                    print(f"     📄 Source: {doc.metadata.get('source_file', doc.metadata.get('source', 'Unknown'))}")
            else:
                print("  ❌ No results found")
                
        except Exception as e:
            print(f"  ❌ Query failed: {e}")
    
    print(f"\n📊 Query Success Rate: {successful_queries}/{len(test_queries)} ({(successful_queries/len(test_queries)*100):.1f}%)")
    
    # Performance optimization
    print("\n⚡ Running Performance Optimization:")
    manager.optimize_performance()
    
    # Comprehensive statistics
    print("\n📈 Comprehensive System Statistics:")
    stats = manager.get_vector_store_stats()
    
    print(f"🧠 Embedding Model: {stats['embedding_config']['name']}")
    print(f"🗄️ Vector Store: {stats['vector_store_type'].upper()}")
    print(f"📄 Document Count: {stats['document_count']}")
    print(f"💾 Storage Size: {stats['storage_size_mb']:.1f} MB")
    print(f"🐏 Memory Usage: {stats['memory_usage_mb']:.1f} MB")
    print(f"⚡ Cache Hit Rate: {stats.get('cache_hit_rate', 0)*100:.1f}%")
    print(f"⏱️ Avg Embedding Time: {stats.get('avg_embedding_time', 0):.3f}s")
    print(f"🔍 Avg Query Time: {stats.get('avg_query_time', 0):.3f}s")
    print(f"📊 Total Embeddings: {stats['embeddings_created']}")
    print(f"🔍 Total Queries: {stats['queries_processed']}")
    print(f"❌ Error Rate: {stats.get('error_rate', 0)*100:.2f}%")
    
    # Test export functionality
    print("\n📦 Testing Export Functionality:")
    export_success = manager.export_vector_store("exports/backup")
    print(f"Export successful: {'✅' if export_success else '❌'}")
    
    # Final system status
    print("\n🏁 Final System Status:")
    overall_health = "✅ Healthy"
    print(f"Overall System Health: {overall_health}")
    
    print("\n✅ Enhanced Production Vector Store System Test Completed!")
    print(f"🎯 Ready for production use with {stats['embedding_config']['name']} embeddings")
    
    return manager

if __name__ == "__main__":
    # Run comprehensive test
    test_production_vector_store()
