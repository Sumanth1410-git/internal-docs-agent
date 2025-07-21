import sys
import platform

print("=== Hardware & Environment Test ===")
print(f"Python version: {sys.version}")
print(f"Platform: {platform.system()} {platform.release()}")
print(f"Processor: {platform.processor()}")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("PyTorch not installed (optional for CPU-only mode)")

try:
    import faiss
    print(f"FAISS version: {faiss.__version__}")
except ImportError:
    print("FAISS not installed - please install faiss-cpu")

try:
    import langchain
    print(f"LangChain version: {langchain.__version__}")
except ImportError:
    print("LangChain not installed")

try:
    import streamlit
    print(f"Streamlit version: {streamlit.__version__}")
except ImportError:
    print("Streamlit not installed")

print("\n=== Memory Check ===")
try:
    import psutil
    memory = psutil.virtual_memory()
    print(f"Total RAM: {memory.total // (1024**3)} GB")
    print(f"Available RAM: {memory.available // (1024**3)} GB")
    print(f"RAM Usage: {memory.percent}%")
except ImportError:
    print("psutil not installed - installing now...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
    import psutil
    memory = psutil.virtual_memory()
    print(f"Total RAM: {memory.total // (1024**3)} GB")
    print(f"Available RAM: {memory.available // (1024**3)} GB")
    print(f"RAM Usage: {memory.percent}%")

print("\n✅ Setup test completed!")
