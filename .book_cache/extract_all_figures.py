import os
import pypdf

pdf_path = r'd:\Documents\IE104.R12\ABUIABA9GAAghIK0ugYowM2h3QY.pdf'
out_dir_en = r'd:\Documents\IE104.R12\distilled_ml_systems_en\figures'
out_dir_vi = r'd:\Documents\IE104.R12\distilled_ml_systems\figures'
os.makedirs(out_dir_en, exist_ok=True)
os.makedirs(out_dir_vi, exist_ok=True)

reader = pypdf.PdfReader(pdf_path)

extracted = []
for p_idx, page in enumerate(reader.pages):
    page_num = p_idx + 1
    for img_idx, img in enumerate(page.images):
        raw_ext = os.path.splitext(img.name)[1]
        ext = raw_ext if raw_ext else '.png'
        filename = f'figure_page_{page_num:03d}_{img_idx+1}{ext}'
        
        path_en = os.path.join(out_dir_en, filename)
        path_vi = os.path.join(out_dir_vi, filename)
        
        with open(path_en, 'wb') as f:
            f.write(img.data)
        with open(path_vi, 'wb') as f:
            f.write(img.data)
            
        extracted.append((page_num, filename, len(img.data)))

print(f'Successfully extracted {len(extracted)} figures into figures/ directories.')
for p, fname, sz in extracted[:10]:
    print(f'  - Page {p}: {fname} ({sz//1024} KB)')

