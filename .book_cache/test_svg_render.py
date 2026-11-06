import subprocess
import os
import re
import markdown

vendor_dir = os.path.abspath(r'd:\Documents\IE104.R12\.book_cache\vendor').replace('\\', '/')

text = r"""# Test Math

Machine Learning is defined as:

$$\text{An engineering approach to } \underbrace{\textbf{learn}}_{(1)} \underbrace{\textbf{complex}}_{(2)} \underbrace{\textbf{patterns}}_{(3)} \text{ from } \underbrace{\textbf{existing data}}_{(4)} \text{ and use these patterns to make } \underbrace{\textbf{predictions}}_{(5)} \text{ on } \underbrace{\textbf{unseen data}}_{(6)}.$$

Mapping function $f: X \to Y$ is too complex. Also lookup table is $O(1)$ complexity.
"""

placeholders = {}
def repl(match):
    key = f"XMATHTOKEN{len(placeholders)}X"
    placeholders[key] = match.group(0)
    return key

# First protect display math, then inline math
protected = re.sub(r"\$\$.*?\$\$", repl, text, flags=re.DOTALL)
protected = re.sub(r"\$[^\$\n]+?\$", repl, protected)

html_body = markdown.markdown(protected, extensions=['extra', 'tables'])
for k, v in placeholders.items():
    html_body = html_body.replace(k, v)

html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <script>
    window.MathJax = {{
      tex: {{
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
      }},
      svg: {{
        fontCache: 'none'
      }}
    }};
    </script>
    <script src="file:///{vendor_dir}/tex-svg.js"></script>
    <style>
        body {{ font-family: Segoe UI, sans-serif; font-size: 11pt; padding: 40px; }}
        mjx-container {{ margin: 12px 0 !important; }}
    </style>
</head>
<body>
    {html_body}
</body>
</html>"""

test_html_path = r'd:\Documents\IE104.R12\.book_cache\test_svg.html'
test_pdf_path = r'd:\Documents\IE104.R12\.book_cache\test_svg.pdf'

with open(test_html_path, 'w', encoding='utf-8') as f:
    f.write(html)

chrome = r'C:\Program Files\Google\Chrome\Application\chrome.exe'
subprocess.run([
    chrome,
    '--headless',
    '--disable-gpu',
    '--no-pdf-header-footer',
    '--virtual-time-budget=5000',
    '--run-all-compositor-stages-before-draw',
    f'--print-to-pdf={test_pdf_path}',
    test_html_path
], check=True)

import pypdf
reader = pypdf.PdfReader(test_pdf_path)
print("Page text:")
print(reader.pages[0].extract_text())

