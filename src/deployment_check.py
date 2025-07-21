import os
import sys
import logging
from pathlib import Path

# Import all our components
from slack_bolt import App
from slack_sdk import WebClient
from rag_agent import ProductionRAGAgent
from vector_store_manager import ProductionVectorStoreManager
from document_loader import OptimizedDocumentLoader

from dotenv import load_dotenv
load_dotenv()

def comprehensive_deployment_check():
    """Comprehensive pre-deployment system validation"""
    print("🔍 **PRE-DEPLOYMENT SYSTEM CHECK**")
    print("=" * 50)
    
    checks_passed = 0
    total_checks = 12
    
    # Check 1: Environment Variables
    print("\n1️⃣ **Environment Variables**")
    required_vars = ['SLACK_BOT_TOKEN', 'SLACK_APP_TOKEN']
    for var in required_vars:
        value = os.getenv(var)
        if value and value != "your_openai_key_here" and len(value) > 10:
            print(f"   ✅ {var}: Configured")
        else:
            print(f"   ❌ {var}: Missing or invalid")
            continue
    checks_passed += 1
    
    # Check 2: Slack Connection Test
    print("\n2️⃣ **Slack API Connection**")
    try:
        client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))
        auth_response = client.auth_test()
        print(f"   ✅ Connected as: {auth_response['user']} ({auth_response['user_id']})")
        print(f"   ✅ Team: {auth_response['team']}")
        checks_passed += 1
    except Exception as e:
        print(f"   ❌ Slack connection failed: {str(e)[:100]}")
    
    # Check 3: Socket Mode Test
    print("\n3️⃣ **Socket Mode Configuration**")
    try:
        app = App(token=os.getenv("SLACK_BOT_TOKEN"))
        from slack_bolt.adapter.socket_mode import SocketModeHandler
        handler = SocketModeHandler(app, os.getenv("SLACK_APP_TOKEN"))
        print("   ✅ Socket Mode handler initialized")
        checks_passed += 1
    except Exception as e:
        print(f"   ❌ Socket Mode failed: {str(e)[:100]}")
    
    # Check 4: Document Loading
    print("\n4️⃣ **Document System**")
    try:
        loader = OptimizedDocumentLoader()
        documents = loader.load_directory("data")
        if documents:
            print(f"   ✅ Loaded {len(documents)} document chunks")
            checks_passed += 1
        else:
            print("   ⚠️ No documents found - will create samples")
            loader.create_sample_documents()
            documents = loader.load_directory("data")
            if documents:
                print(f"   ✅ Created {len(documents)} sample document chunks")
                checks_passed += 1
    except Exception as e:
        print(f"   ❌ Document loading failed: {str(e)[:100]}")
    
    # Check 5: Vector Store System
    print("\n5️⃣ **Vector Store System**")
    try:
        vector_manager = ProductionVectorStoreManager()
        embedding_ready = vector_manager.initialize_embedding_service()
        if embedding_ready:
            print(f"   ✅ Embedding service: {vector_manager.embedding_config['name']}")
            checks_passed += 1
        else:
            print("   ❌ Embedding service initialization failed")
    except Exception as e:
        print(f"   ❌ Vector store failed: {str(e)[:100]}")
    
    # Check 6: RAG Agent
    print("\n6️⃣ **RAG Agent System**")
    try:
        rag_agent = ProductionRAGAgent()
        status = rag_agent.initialize_components()
        if status['vector_store'] and status['embeddings']:
            print("   ✅ RAG components initialized successfully")
            checks_passed += 1
        else:
            print(f"   ⚠️ RAG status: {status}")
    except Exception as e:
        print(f"   ❌ RAG agent failed: {str(e)[:100]}")
    
    # Check 7: GPU/Hardware
    print("\n7️⃣ **Hardware Configuration**")
    try:
        import torch
        import psutil
        
        print(f"   ✅ CPU Cores: {psutil.cpu_count()}")
        print(f"   ✅ RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        
        if torch.cuda.is_available():
            print(f"   ✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"   ✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        else:
            print("   ⚠️ GPU: Not available (CPU mode)")
        checks_passed += 1
    except Exception as e:
        print(f"   ❌ Hardware check failed: {str(e)[:100]}")
    
    # Check 8: File Permissions
    print("\n8️⃣ **File System Permissions**")
    try:
        test_dirs = ['cache', 'vectorstore', 'data', 'exports']
        for dir_name in test_dirs:
            Path(dir_name).mkdir(exist_ok=True)
            test_file = Path(dir_name) / "test.txt"
            test_file.write_text("test")
            test_file.unlink()
        print("   ✅ File system permissions OK")
        checks_passed += 1
    except Exception as e:
        print(f"   ❌ File permissions failed: {str(e)[:100]}")
    
    # Check 9: Memory Availability
    print("\n9️⃣ **Memory Availability**")
    try:
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        if available_gb > 2:
            print(f"   ✅ Available RAM: {available_gb:.1f} GB")
            checks_passed += 1
        else:
            print(f"   ⚠️ Low memory: {available_gb:.1f} GB available")
    except Exception as e:
        print(f"   ❌ Memory check failed: {str(e)[:100]}")
    
    # Check 10: Network Connectivity
    print("\n🔟 **Network Connectivity**")
    try:
        import requests
        response = requests.get("https://slack.com", timeout=5)
        if response.status_code == 200:
            print("   ✅ Internet connectivity OK")
            checks_passed += 1
        else:
            print(f"   ❌ Network issues: Status {response.status_code}")
    except Exception as e:
        print(f"   ❌ Network check failed: {str(e)[:100]}")
    
    # Check 11: Dependencies
    print("\n1️⃣1️⃣ **Package Dependencies**")
    try:
        critical_packages = [
            'slack_sdk', 'slack_bolt', 'langchain', 'langchain_openai',
            'faiss', 'sentence_transformers', 'numpy', 'psutil'
        ]
        
        for package in critical_packages:
            __import__(package.replace('-', '_'))
        print(f"   ✅ All critical packages available")
        checks_passed += 1
    except ImportError as e:
        print(f"   ❌ Missing package: {str(e)}")
    
    # Check 12: Query Processing Test
    print("\n1️⃣2️⃣ **End-to-End Query Test**")
    try:
        rag_agent = ProductionRAGAgent()
        rag_agent.initialize_components()
        
        test_result = rag_agent.query_with_context(
            "What's our refund policy?",
            query_type="deployment_test"
        )
        
        if test_result['answer'] and len(test_result['answer']) > 20:
            print(f"   ✅ Query processing: {test_result['response_time']:.2f}s")
            print(f"   ✅ Answer length: {len(test_result['answer'])} chars")
            checks_passed += 1
        else:
            print("   ❌ Query test failed - no valid response")
    except Exception as e:
        print(f"   ❌ Query test failed: {str(e)[:100]}")
    
    # Final Results
    print("\n" + "=" * 50)
    print(f"📊 **DEPLOYMENT READINESS SCORE: {checks_passed}/{total_checks}**")
    
    if checks_passed >= 10:
        print("🟢 **READY FOR DEPLOYMENT** - All critical systems operational")
        return True
    elif checks_passed >= 8:
        print("🟡 **DEPLOYMENT WITH CAUTION** - Some issues detected")
        return True
    else:
        print("🔴 **NOT READY FOR DEPLOYMENT** - Critical issues must be resolved")
        return False

if __name__ == "__main__":
    comprehensive_deployment_check()
