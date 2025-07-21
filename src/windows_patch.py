import sys
import platform
from types import ModuleType

def patch_pwd_for_windows():
    """Patch pwd module for Windows compatibility"""
    if platform.system() == "Windows" and 'pwd' not in sys.modules:
        # Create a mock pwd module
        mock_pwd = ModuleType('pwd')
        
        # Mock the required functions
        class MockPasswordEntry:
            def __init__(self, pw_name="unknown"):
                self.pw_name = pw_name
        
        def mock_getpwuid(uid):
            return MockPasswordEntry()
        
        mock_pwd.getpwuid = mock_getpwuid
        sys.modules['pwd'] = mock_pwd

# Apply patch before importing langchain
patch_pwd_for_windows()
