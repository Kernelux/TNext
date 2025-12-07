import fitz
import os

ref_path = 'ref'
papers = []

for pdf_file in sorted(os.listdir(ref_path)):
    if pdf_file.endswith('.pdf'):
        doc = fitz.open(os.path.join(ref_path, pdf_file))
        text = ''
        # Read first 8 pages to get abstract and key contributions
        for page_num in range(min(8, len(doc))):
            text += doc[page_num].get_text()
        papers.append((pdf_file, text))
        doc.close()

for name, content in papers:
    print(f'\n{"="*80}')
    print(f'PAPER: {name}')
    print(f'{"="*80}')
    print(content[:7000])
    print('...[truncated]...')
