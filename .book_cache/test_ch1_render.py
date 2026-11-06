import subprocess
import os
import re
import markdown

vendor_dir = os.path.abspath(r'd:\Documents\IE104.R12\.book_cache\vendor').replace('\\', '/')
figures_dir = os.path.abspath(r'd:\Documents\IE104.R12\distilled_ml_systems_en\figures').replace('\\', '/')

with open(r'd:\Documents\IE104.R12\distilled_ml_systems_en\Chapter_01_ML_Systems_in_Production.md', 'r', encoding='utf-8') as f:
    raw_md = f.read()

def fix_list_indentation(md_text):
    lines = md_text.splitlines()
    fixed = []
    in_numbered_list = False
    for line in lines:
        if re.match(r'^\s*\d+\.\s+', line):
            in_numbered_list = True
            fixed.append(line)
        elif in_numbered_list and re.match(r'^\s{1,3}[-*+]\s+', line):
            stripped = line.lstrip()
            fixed.append('    ' + stripped)
        else:
            if not line.startswith(' ') and line.strip() != '':
                in_numbered_list = False
            fixed.append(line)
    return '\n'.join(fixed)

def fix_img(match):
    alt = match.group(1)
    rel_path = match.group(2)
    base = os.path.basename(rel_path)
    abs_img_path = os.path.join(figures_dir, base).replace('\\', '/')
    return f'<div class="figure-wrapper"><img src="file:///{abs_img_path}" alt="{alt}"/><div class="figure-caption">{alt}</div></div>'

fixed_md = fix_list_indentation(raw_md)
fixed_md = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', fix_img, fixed_md)

md = markdown.Markdown(extensions=['extra', 'tables', 'fenced_code', 'toc'])
ch_html = md.convert(fixed_md)

# Convert mermaid code blocks
ch_html = re.sub(
    r'<pre><code class="language-mermaid">(.*?)</code></pre>',
    r'<pre class="mermaid">\1</pre>',
    ch_html,
    flags=re.DOTALL
)

html_content = f"""<!DOCTYPE html>
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
    <script src="file:///{vendor_dir}/mermaid.min.js"></script>
    <script>
    mermaid.initialize({{
        startOnLoad: true,
        theme: 'neutral',
        flowchart: {{ useMaxWidth: false, htmlLabels: true }}
    }});
    </script>
    <style>
        @page {{
            size: A4;
            margin: 20mm 15mm 20mm 15mm;
        }}
        body {{
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, Arial, sans-serif;
            color: #1a202c;
            line-height: 1.6;
            font-size: 10pt;
            background: #fff;
        }}
        h1 {{ font-size: 18pt; border-bottom: 2px solid #2563eb; padding-bottom: 6px; color: #0f172a; }}
        h2 {{ font-size: 14pt; border-bottom: 1px solid #e2e8f0; padding-bottom: 4px; color: #1e293b; margin-top: 20px; }}
        h3 {{ font-size: 11.5pt; color: #334155; margin-top: 16px; }}
        mjx-container {{ margin: 10px 0 !important; }}
        mjx-container[display="true"] {{ text-align: center; margin: 14px 0 !important; }}
        pre.mermaid {{
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 14px;
            margin: 16px 0;
            text-align: center;
        }}
        pre.mermaid svg {{
            max-width: 100% !important;
            height: auto !important;
        }}
        .figure-wrapper {{ text-align: center; margin: 18px 0; }}
        .figure-wrapper img {{ max-width: 90%; max-height: 380px; border: 1px solid #e2e8f0; border-radius: 4px; }}
        .figure-caption {{ font-size: 8.5pt; color: #64748b; font-style: italic; margin-top: 4px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 14px 0; font-size: 9pt; }}
        th, td {{ border: 1px solid #cbd5e1; padding: 6px 9px; vertical-align: top; }}
        th {{ background: #f1f5f9; font-weight: 600; }}
        tr:nth-child(even) {{ background: #f8fafc; }}
        code {{ font-family: Consolas, monospace; font-size: 8.5pt; background: #f1f5f9; padding: 2px 4px; border-radius: 3px; }}
        pre {{ background: #0f172a; color: #f8fafc; padding: 10px; border-radius: 6px; font-size: 8pt; }}
        blockquote {{ border-left: 4px solid #2563eb; background: #f8fafc; padding: 8px 14px; margin: 12px 0; font-size: 9.5pt; }}
        ol, ul {{ padding-left: 20px; margin-bottom: 8px; }}
        li {{ margin-bottom: 3px; }}
    </style>
</head>
<body>
    {ch_html}
</body>
</html>"""

test_html_path = r'd:\Documents\IE104.R12\.book_cache\test_ch1.html'
test_pdf_path = r'd:\Documents\IE104.R12\.book_cache\test_ch1.pdf'

with open(test_html_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

chrome = r'C:\Program Files\Google\Chrome\Application\chrome.exe'
subprocess.run([
    chrome,
    '--headless',
    '--disable-gpu',
    '--no-pdf-header-footer',
    '--virtual-time-budget=8000',
    '--run-all-compositor-stages-before-draw',
    f'--print-to-pdf={test_pdf_path}',
    test_html_path
], check=True)

print("Test Ch1 PDF generated, size:", os.path.getsize(test_pdf_path))

