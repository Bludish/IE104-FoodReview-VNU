import subprocess
import os

test_html = """<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <script>mermaid.initialize({startOnLoad:true});</script>
</head>
<body>
    <h1>Mermaid Diagram Test</h1>
    <pre class="mermaid">
    flowchart LR
        A[Data Source] --> B[Feature Store]
        B --> C[Model Training]
        C --> D[Model Serving]
    </pre>
</body>
</html>"""

html_path = r'd:\Documents\IE104.R12\.book_cache\test_mermaid.html'
pdf_path = r'd:\Documents\IE104.R12\.book_cache\test_mermaid.pdf'

with open(html_path, 'w', encoding='utf-8') as f:
    f.write(test_html)

chrome = r'C:\Program Files\Google\Chrome\Application\chrome.exe'
subprocess.run([
    chrome,
    '--headless',
    '--disable-gpu',
    '--virtual-time-budget=5000',
    '--run-all-compositor-stages-before-draw',
    f'--print-to-pdf={pdf_path}',
    html_path
], check=True)

print('Test PDF generated, size:', os.path.getsize(pdf_path))

