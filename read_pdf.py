import sys
import io

try:
    from pypdf import PdfReader
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pypdf", "--user"])
    from pypdf import PdfReader

import sys
# Set output encoding to utf-8 to avoid charmap errors
sys.stdout.reconfigure(encoding='utf-8')

try:
    reader = PdfReader(r"C:\_RGS_2026\5 semestre\ING SOLU CON IA_008V_OLS\Evaluacion 1\Rubrica.pdf")
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
        
    print("=== PDF CONTENT START ===")
    print(text)
    print("=== PDF CONTENT END ===")
except Exception as e:
    print(f"Error reading PDF: {e}")
