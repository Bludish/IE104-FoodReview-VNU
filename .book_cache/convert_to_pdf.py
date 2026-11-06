import os
import re
import subprocess
import markdown

chapters_dir = r'd:\Documents\IE104.R12\distilled_ml_systems'
figures_dir = os.path.abspath(r'd:\Documents\IE104.R12\distilled_ml_systems\figures').replace('\\', '/')
vendor_dir = os.path.abspath(r'd:\Documents\IE104.R12\.book_cache\vendor').replace('\\', '/')

output_html = r'd:\Documents\IE104.R12\.book_cache\full_book_vi.html'
output_pdf_root = r'd:\Documents\IE104.R12\Designing_Machine_Learning_Systems_Distilled.pdf'
output_pdf_dist = r'd:\Documents\IE104.R12\distilled_ml_systems\Designing_Machine_Learning_Systems_Distilled.pdf'

chapter_files = [
    'Chapter_01_ML_Systems_in_Production.md',
    'Chapter_02_Data_Engineering_Fundamentals.md',
    'Chapter_03_Training_Data.md',
    'Chapter_04_Feature_Engineering.md',
    'Chapter_05_Model_Development.md',
    'Chapter_06_Model_Deployment.md',
    'Chapter_07_Why_ML_Systems_Fail_in_Production.md',
]

css = """
@page {
    size: A4;
    margin: 20mm 15mm 20mm 15mm;
    @bottom-center {
        content: counter(page);
    }
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    color: #0f172a;
    line-height: 1.65;
    font-size: 10.5pt;
    background: #fff;
    margin: 0;
    padding: 0;
    -webkit-print-color-adjust: exact;
}

.cover-page {
    page-break-after: always;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    min-height: 90vh;
    text-align: center;
    padding: 60px 20px;
}

.cover-title {
    font-size: 32pt;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 14px;
    line-height: 1.2;
}

.cover-subtitle {
    font-size: 15pt;
    font-weight: 600;
    color: #2563eb;
    margin-bottom: 26px;
    letter-spacing: 0.5px;
}

.cover-author {
    font-size: 12pt;
    color: #475569;
    margin-bottom: 40px;
}

.cover-meta {
    font-size: 10pt;
    color: #64748b;
    border-top: 1px solid #e2e8f0;
    padding-top: 24px;
    max-width: 540px;
    line-height: 1.75;
}

.chapter-container {
    page-break-before: always;
    padding-top: 10px;
}

h1 {
    font-size: 20pt;
    font-weight: 800;
    color: #0f172a;
    border-bottom: 2.5px solid #2563eb;
    padding-bottom: 8px;
    margin-top: 28px;
    margin-bottom: 18px;
    page-break-after: avoid;
}

h2 {
    font-size: 15pt;
    font-weight: 700;
    color: #1e293b;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 6px;
    margin-top: 24px;
    margin-bottom: 14px;
    page-break-after: avoid;
}

h3 {
    font-size: 12.5pt;
    font-weight: 600;
    color: #334155;
    margin-top: 20px;
    margin-bottom: 10px;
    page-break-after: avoid;
}

h4 {
    font-size: 11pt;
    font-weight: 600;
    color: #475569;
    margin-top: 16px;
    margin-bottom: 8px;
    page-break-after: avoid;
}

p {
    margin-top: 0;
    margin-bottom: 12px;
    text-align: justify;
}

ul, ol {
    margin-top: 0;
    margin-bottom: 12px;
    padding-left: 24px;
}

li {
    margin-bottom: 5px;
}

blockquote {
    border-left: 4px solid #2563eb;
    background-color: #f8fafc;
    padding: 12px 18px;
    margin: 14px 0;
    color: #334155;
    font-size: 10pt;
    border-radius: 0 6px 6px 0;
}

blockquote p:last-child {
    margin-bottom: 0;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 18px 0;
    font-size: 9.5pt;
    page-break-inside: avoid;
}

th, td {
    border: 1px solid #cbd5e1;
    padding: 8px 12px;
    text-align: left;
    vertical-align: top;
}

th {
    background-color: #f1f5f9;
    color: #0f172a;
    font-weight: 600;
}

tr:nth-child(even) {
    background-color: #f8fafc;
}

code {
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 9pt;
    background-color: #f1f5f9;
    color: #0f172a;
    padding: 2px 5px;
    border-radius: 4px;
    border: 1px solid #e2e8f0;
}

pre {
    background-color: #0f172a;
    color: #f8fafc;
    padding: 14px 16px;
    border-radius: 6px;
    overflow-x: auto;
    font-size: 8.5pt;
    line-height: 1.45;
    margin: 14px 0;
    page-break-inside: avoid;
}

pre code {
    background-color: transparent;
    color: inherit;
    padding: 0;
    border: none;
    font-size: inherit;
}

.figure-wrapper {
    text-align: center;
    margin: 22px 0;
    page-break-inside: avoid;
}

.figure-wrapper img {
    max-width: 92%;
    max-height: 440px;
    border-radius: 6px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}

.figure-caption {
    font-size: 9pt;
    color: #64748b;
    margin-top: 8px;
    font-style: italic;
}

mjx-container {
    font-size: 105% !important;
}

mjx-container[display="true"] {
    text-align: center;
    margin: 16px 0 !important;
}

pre.mermaid {
    background-color: #f8fafc;
    color: #0f172a;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    text-align: center;
    padding: 16px;
    margin: 20px 0;
    page-break-inside: avoid;
}

pre.mermaid svg {
    max-width: 98% !important;
    height: auto !important;
}

hr {
    border: 0;
    height: 1px;
    background: #cbd5e1;
    margin: 28px 0;
}
"""

cover_html = """
<div class="cover-page">
    <div style="font-size: 11pt; text-transform: uppercase; letter-spacing: 2px; color: #64748b; margin-bottom: 20px;">
        BẢN TINH GỌN CHUYÊN SÂU — DISTILLED MASTER NOTES
    </div>
    <div class="cover-title">
        Designing Machine Learning Systems
    </div>
    <div class="cover-subtitle">
        An Iterative Process for Production-Ready Applications
    </div>
    <div class="cover-author">
        <strong>Tác giả gốc</strong>: Chip Huyen (O'Reilly Media) &bull; <strong>Biên soạn tinh giản</strong>: AI Multi-Agent System
    </div>
    <div class="cover-meta">
        <strong>Nguyên tắc biên soạn</strong>:<br/>
        &bull; <strong>100% Cốt lõi học thuật</strong>: Công thức toán học vector SVG, kiến trúc hệ thống, thuật ngữ chuyên ngành chuẩn xác.<br/>
        &bull; <strong>Triệt tiêu râu ria</strong>: Loại bỏ hoàn toàn chuyện phiếm, lời tự sự và các ví dụ đời thường kéo dài.<br/>
        &bull; <strong>Trực quan hóa</strong>: Toàn bộ bảng biểu đối chiếu và lưu đồ kiến trúc Mermaid được vẽ đồ họa trực quan.<br/>
        &bull; <strong>Quy mô</strong>: 7 Chương toàn diện &bull; Cẩm nang thực chiến Production.
    </div>
</div>
"""

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

def process_markdown_and_protect_math(raw_text):
    text = fix_list_indentation(raw_text)

    def fix_img(match):
        alt = match.group(1)
        rel_path = match.group(2)
        base = os.path.basename(rel_path)
        abs_img_path = os.path.join(figures_dir, base).replace('\\', '/')
        return f'<div class="figure-wrapper"><img src="file:///{abs_img_path}" alt="{alt}"/><div class="figure-caption">{alt}</div></div>'

    text = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', fix_img, text)

    math_tokens = {}
    def save_math(match):
        token = f"@@@MATHTOKEN{len(math_tokens)}@@@"
        math_tokens[token] = match.group(0)
        return token

    text = re.sub(r'\$\$.*?\$\$', save_math, text, flags=re.DOTALL)
    text = re.sub(r'\$[^\$\n]+?\$', save_math, text)

    md = markdown.Markdown(extensions=['extra', 'tables', 'fenced_code', 'toc'])
    html = md.convert(text)

    for token, math_content in math_tokens.items():
        html = html.replace(token, math_content)

    html = re.sub(
        r'<pre><code class="language-mermaid">(.*?)</code></pre>',
        r'<pre class="mermaid">\1</pre>',
        html,
        flags=re.DOTALL
    )

    return html

print("Reading and rendering Vietnamese chapters with protected Math...")
rendered_chapters = []

for filename in chapter_files:
    file_path = os.path.join(chapters_dir, filename)
    if not os.path.exists(file_path):
        continue
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    ch_html = process_markdown_and_protect_math(raw_text)
    rendered_chapters.append(f'<div class="chapter-container">{ch_html}</div>')
    print(f"Rendered {filename} successfully.")

full_html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <title>Designing Machine Learning Systems - Chip Huyen (Bản Tinh Gọn)</title>
    <style>{css}</style>
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
</head>
<body>
    {cover_html}
    {''.join(rendered_chapters)}
</body>
</html>
"""

with open(output_html, 'w', encoding='utf-8') as f:
    f.write(full_html)

print(f"Full HTML written to {output_html} ({os.path.getsize(output_html)//1024} KB)")

chrome_path = r'C:\Program Files\Google\Chrome\Application\chrome.exe'
print("Running Chrome Headless PDF generation for Vietnamese edition...")
subprocess.run([
    chrome_path,
    '--headless',
    '--disable-gpu',
    '--no-pdf-header-footer',
    '--virtual-time-budget=10000',
    '--run-all-compositor-stages-before-draw',
    f'--print-to-pdf={output_pdf_root}',
    output_html
], check=True)

if os.path.exists(output_pdf_root):
    size_mb = os.path.getsize(output_pdf_root) / (1024 * 1024)
    print(f"SUCCESS: Generated Vietnamese Master PDF at {output_pdf_root} ({size_mb:.2f} MB)")
    import shutil
    shutil.copy2(output_pdf_root, output_pdf_dist)
    print(f"Copied PDF to {output_pdf_dist}")
