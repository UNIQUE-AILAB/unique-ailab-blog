"""One-time import of the author's PDF exports into editable HTML articles.

Requires PyMuPDF. Prose and code become HTML text; original figures and display
equations are preserved as images so mathematical layout is not lost.
"""
import html
import json
import re
from pathlib import Path

import fitz

HERE = Path(__file__).resolve().parent


def clean(text):
    return text.replace('\u200b', '').replace('\u00a0', ' ')


def convert(note):
    slug = note['path'].strip('/').split('/')[-1]
    assets = HERE / 'note-images' / slug
    assets.mkdir(parents=True, exist_ok=True)
    document = fitz.open(HERE / note['source'])
    items = []
    image_count = 0
    for page in document:
        blocks = page.get_text('dict')['blocks']
        links = [l for l in page.get_links() if l.get('uri')]
        code_regions = [d['rect'] for d in page.get_drawings()
                        if d.get('fill') and d['rect'].width > 400
                        and d['rect'].height > 20
                        and .93 < min(d['fill']) < .99
                        and max(d['fill']) - min(d['fill']) < .04]
        math_rects = []
        for block in blocks:
            if block['type'] == 0 and any('KaTeX' in s['font'] for l in block['lines'] for s in l['spans']):
                math_spans = [s for l in block['lines'] for s in l['spans'] if 'KaTeX' in s['font']]
                rect = fitz.Rect(math_spans[0]['bbox'])
                for span in math_spans[1:]:
                    rect.include_rect(span['bbox'])
                # Fractions, summations and subscripts are often separate blocks.
                for previous in math_rects:
                    if (previous + (-3, -4, 3, 4)).intersects(rect):
                        previous.include_rect(rect)
                        break
                else:
                    math_rects.append(rect)
        # A second union handles a fragment connecting two earlier groups.
        changed = True
        while changed:
            changed = False
            for i in range(len(math_rects)):
                for j in range(i + 1, len(math_rects)):
                    if (math_rects[i] + (-3, -4, 3, 4)).intersects(math_rects[j]):
                        math_rects[i].include_rect(math_rects.pop(j))
                        changed = True
                        break
                if changed:
                    break
        events = []
        inline = []
        ordinary_lines = [l for b in blocks if b['type'] == 0 for l in b['lines']
                          if len(clean(''.join(s['text'] for s in l['spans'] if 'KaTeX' not in s['font'])).strip()) > 20]
        for rect in math_rects:
            if any(fitz.Rect(l['bbox']).intersects(rect) for l in ordinary_lines):
                image_count += 1
                filename = f'formula-{image_count:02d}.png'
                clip = (rect + (-1, -1, 1, 1)) & page.rect
                page.get_pixmap(matrix=fitz.Matrix(3, 3), clip=clip, alpha=False).save(assets / filename)
                markup = f'<img class="note-inline-formula" src="/assets/note-images/{slug}/{filename}" alt="公式" style="width:{round(clip.width * 1.333)}px">'
                inline.append((rect, markup))
            else:
                events.append((rect.y0, 'formula', rect))
        numbers = []
        for block in blocks:
            if block['type'] == 1:
                events.append((block['bbox'][1], 'image', block))
                continue
            for line in block['lines']:
                spans = [s for s in line['spans'] if 'KaTeX' not in s['font']]
                if not spans:
                    continue
                text = clean(''.join(s['text'] for s in spans))
                mono = any('SourceCodePro' in s['font'] for s in spans)
                rect = fitz.Rect(line['bbox'])
                # Chinese-only wrapped comments use a fallback font, but remain
                # part of the code block's shaded rectangle.
                mono = mono or (rect.x0 >= 65 and any(r.y0 <= (rect.y0 + rect.y1) / 2 <= r.y1 for r in code_regions))
                if mono and rect.x1 < 65 and text.strip().isdigit():
                    numbers.append((rect.y0, text.strip()))
                    continue
                if text.strip() == '代码块':
                    continue
                if mono:
                    events.append((rect.y0, 'code', (text, rect)))
                    continue
                if not text.strip():
                    continue
                size = max(s['size'] for s in spans)
                if size >= 24:  # The title is supplied by the article template.
                    continue
                rendered = []
                for span in spans:
                    value = html.escape(clean(span['text']))
                    box = fitz.Rect(span['bbox'])
                    link = next((l for l in links if l['from'].contains(fitz.Point((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2))), None)
                    if link and value.strip():
                        value = f'<a href="{html.escape(link["uri"], quote=True)}">{value}</a>'
                    rendered.append((box.x0, value))
                for math_rect, markup in inline:
                    if rect.intersects(math_rect):
                        rendered.append((math_rect.x0, markup))
                rendered.sort(key=lambda item: item[0])
                events.append((rect.y0, 'heading' if size >= 15 else 'text', (''.join(v for _, v in rendered), rect, size)))
        events.sort(key=lambda e: e[0])
        previous_bottom = None
        for y, kind, value in events:
            if kind in ('image', 'formula'):
                image_count += 1
                if kind == 'image':
                    ext = value['ext']
                    filename = f'figure-{image_count:02d}.{ext}'
                    (assets / filename).write_bytes(value['image'])
                    width = value['bbox'][2] - value['bbox'][0]
                else:
                    filename = f'formula-{image_count:02d}.png'
                    rect = (value + (-2, -2, 2, 2)) & page.rect
                    page.get_pixmap(matrix=fitz.Matrix(3, 3), clip=rect, alpha=False).save(assets / filename)
                    width = rect.width
                alt = '公式' if kind == 'formula' else '原文配图'
                markup = f'<figure class="note-{kind}"><img src="/assets/note-images/{slug}/{filename}" alt="{alt}" loading="lazy" style="width:{round(width * 1.333)}px"></figure>'
                items.append(['html', markup])
                previous_bottom = None
            elif kind == 'code':
                text, rect = value
                numbered = any(abs(ny - y) < 3 for ny, _ in numbers)
                if items and items[-1][0] == 'code':
                    # A wrapped visual line has no line number in the export.
                    if numbered:
                        items[-1][1] += '\n' + text
                    else:
                        items[-1][1] += text.lstrip()
                else:
                    items.append(['code', text])
                previous_bottom = None
            elif kind == 'heading':
                text, rect, size = value
                tag = 'h2' if size >= 19 else 'h3'
                items.append(['html', f'<{tag}>{text}</{tag}>'])
                previous_bottom = None
            else:
                text, rect, size = value
                # Rejoin visual line wraps without retaining PDF page breaks.
                if items and items[-1][0] == 'paragraph' and (previous_bottom is None or rect.y0 - previous_bottom < 9):
                    separator = ' ' if re.search(r'[A-Za-z0-9]$', items[-1][1]) and re.match(r'^[A-Za-z0-9]', text) else ''
                    items[-1][1] += separator + text
                else:
                    items.append(['paragraph', text])
                previous_bottom = rect.y1
    result = []
    for kind, content in items:
        if kind == 'code':
            content = '\n'.join(line.rstrip() for line in content.split('\n'))
            result.append('<pre><code class="language-python">' + html.escape(content) + '</code></pre>')
        elif kind == 'paragraph':
            result.append('<p>' + content + '</p>')
        else:
            result.append(content)
    target = HERE / 'articles' / f'{slug}.html'
    target.parent.mkdir(exist_ok=True)
    target.write_text('\n'.join(result) + '\n', encoding='utf-8')
    print(f'{slug}: {len(document)} source pages, {image_count} figures/formulas, {sum(i[0] == "code" for i in items)} code blocks')


if __name__ == '__main__':
    for note in json.loads((HERE / 'notes.json').read_text(encoding='utf-8')):
        if note['source'].startswith('pdfs/'):
            convert(note)
