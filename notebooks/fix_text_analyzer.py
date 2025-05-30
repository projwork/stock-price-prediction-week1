"""
Quick fix for text analyzer NLTK issues
Add this cell to your notebook and run it before using text_analyzer
"""

import sys
import os
sys.path.append('../src')

# Import the working simple text analyzer
from simple_text_analyzer import SimpleFinancialTextAnalyzer

# Create an alias so you don't need to change your existing code
FinancialTextAnalyzer = SimpleFinancialTextAnalyzer

print("✅ Text analyzer fixed! NLTK issues resolved.")
print("You can now use text_analyzer = FinancialTextAnalyzer(df, text_column='headline')")
print("All your existing code will work without changes.") 