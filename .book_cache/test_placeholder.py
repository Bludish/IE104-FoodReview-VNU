import re
import markdown

text = r"""$$\text{An engineering approach to } \underbrace{\textbf{learn}}_{(1)} \underbrace{\textbf{complex}}_{(2)}$$"""
print("Original:", repr(text))

placeholders = {}
def repl(match):
    key = f"XMATHTOKEN{len(placeholders)}X"
    placeholders[key] = match.group(0)
    return key

# First protect display math, then inline math
protected = re.sub(r"\$\$.*?\$\$", repl, text, flags=re.DOTALL)
protected = re.sub(r"\$[^\$\n]+?\$", repl, protected)

print("Protected:", protected)
html = markdown.markdown(protected, extensions=['extra', 'tables'])
for k, v in placeholders.items():
    html = html.replace(k, v)

print("Restored HTML:", html)

