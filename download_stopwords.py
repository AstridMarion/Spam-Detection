"""Script to download NLTK stopwords with SSL workaround for macOS."""
import ssl
import nltk

# Workaround for SSL certificate issue on macOS
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download stopwords
print("📥 Downloading NLTK stopwords...")
nltk.download('stopwords')
print("✅ Stopwords downloaded successfully!")

# Test that it works
from nltk.corpus import stopwords
words = stopwords.words('english')
print(f"✅ Loaded {len(words)} English stopwords")
print(f"First 10 stopwords: {words[:10]}")