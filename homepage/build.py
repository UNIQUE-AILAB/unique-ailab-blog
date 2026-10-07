"""Add research placeholders to the original website, preserving its theme."""
import argparse
import html
import json
import os
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = 'https://github.com/AetherNoah/UNIQUE-AILAB.github.io'
escape = html.escape


def belongs_to(note, slug):
    return note['direction'] == slug or slug in note.get('related_directions', [])


def research_navigation(source, directions):
    if 'assets/homepage.css' not in source:
        source = source.replace('</head>', '<link rel="stylesheet" href="/assets/homepage.css">\n</head>', 1)
    entries = ''.join(f'<li><a class="category-link" href="/research/{d["slug"]}/index.html">{escape(d["name"])}</a></li>' for d in directions)
    menu = f'<li class="active research-navigation"><a href="/index.html#research">研究方向</a><ul class="submenu">{entries}</ul></li>'
    # Replace the old category menu as well as an already-generated research menu.
    source = re.sub(r'<li>\s*<a href="[^\"]*index.html#research">研究方向</a>\s*</li>\s*', '', source)
    source = re.sub(r'<li class="(?:active(?: research-navigation)?|research-navigation)">\s*<a href="[^\"]*">(?:分类|研究方向)</a>\s*<ul class="submenu">.*?</ul>\s*</li>', menu, source, count=1, flags=re.S)
    return source


def prepare(template, directions):
    template = template.replace('<html>', '<html lang="zh-CN">', 1)
    template = template.replace('</head>', '<link rel="stylesheet" href="/assets/homepage.css">\n</head>', 1)
    template = research_navigation(template, directions)
    template = template.replace('href="https://github.com/UNIQUE-AILAB"', f'href="{REPO}"')
    return '\n'.join(line.rstrip().expandtabs(4) for line in template.split('\n'))


def home(template, directions, notes):
    cards = []
    common_count = sum(note['direction'] == 'common' for note in notes)
    for direction in directions:
        count = sum(belongs_to(note, direction['slug']) for note in notes)
        cover = direction['cover']
        cards.append(f'''<article class="link_box special research-card">
<a class="research-cover research-cover--{escape(cover['kind'])}" href="/research/{direction['slug']}/index.html" tabindex="-1" aria-hidden="true">
<img src="/assets/research-covers/{escape(cover['file'])}" alt="" loading="lazy" decoding="async" width="640" height="280"></a>
<div class="research-card-body">
<h3><a href="/research/{direction['slug']}/index.html">{escape(direction['name'])}</a></h3>
<p class="research-english">{escape(direction['english'])}</p>
<p>{escape(direction['summary'])}</p>
<p class="research-status">{str(count) + ' 篇方向笔记' if count else '方向笔记待补充'} · 通用基础笔记 {common_count} 篇</p>
<a class="button" href="/research/{direction['slug']}/index.html#notes">浏览笔记</a>
<a class="research-credit" href="{escape(cover['source'], quote=True)}" target="_blank" rel="noopener noreferrer" aria-label="封面来源：{escape(cover['alt'])}">{escape(cover['label'])} ↗</a></div></article>''')
    section = f'''<section id="research" class="research-area" aria-labelledby="research-title">
<header class="link_box special research-heading"><h2 id="research-title">研究方向</h2>
<p>按研究方向浏览团队的技术笔记与学习记录。</p></header>
<div class="research-grid">{''.join(cards)}</div></section>\n'''
    # Move the original article collection into the direction note lists.
    pattern = r'<!-- 文章列表   s -->.*?<!-- 文章列表   end -->'
    result, count = re.subn(pattern, lambda m: section, template, count=1, flags=re.S)
    assert count == 1
    return result


def note_list(notes):
    items = []
    for note in notes:
        author = f' · {escape(note["author"])}' if note['author'] else ''
        date_label = escape(note.get('date_label', ''))
        pdf_link = f' · <a href="/{escape(note["pdf"])}" download="{escape(note["title"], quote=True)}.pdf">下载 PDF（{note["pages"]} 页）</a>' if 'pdf' in note else ''
        items.append(f'''<li class="research-note"><h3><a href="/{escape(note['path'], quote=True)}">{escape(note['title'])}</a></h3>
<p class="research-status">{date_label} <time datetime="{note['date']}">{note['date']}</time>{author}</p>
<p>{escape(note['summary'])}</p><a href="/{escape(note['path'], quote=True)}">阅读全文 →</a>{pdf_link}</li>''')
    return '<ul class="research-notes">' + ''.join(items) + '</ul>'


def detail(template, direction, notes):
    head = template.split('</head>', 1)[0] + '</head>'
    head = re.sub(r'<title>.*?</title>', f'<title>{escape(direction["name"])} · Unique AI Lab</title>', head, flags=re.S)
    nav = template.split('<nav id="nav"', 1)[1].split('</nav>', 1)[0]
    nav = '<nav id="nav"' + nav + '</nav>'
    selected = [note for note in notes if belongs_to(note, direction['slug'])]
    common = [note for note in notes if note['direction'] == 'common']
    content = note_list(selected) if selected else '<p class="research-status">暂无本方向笔记，后续持续补充。</p>'
    topics = ''.join(f'<li>{escape(topic)}</li>' for topic in direction['topics'])
    return f'''{head}<body class="is-loading"><div id="wrapper" class="fade-in">
<header id="header"><a href="/index.html" class="logo">UNIQUE AI</a></header>{nav}
<main id="main"><article class="research-detail"><header><h1>{escape(direction['name'])}</h1>
<p>{escape(direction['english'])}</p></header><p>{escape(direction['summary'])}</p>
<ul>{topics}</ul>
<section class="research-slot" id="notes"><h2>技术笔记</h2>{content}</section>
<section class="research-slot" id="common-notes"><h2>通用基础笔记</h2><p>各方向共用的机器学习基础知识。</p>{note_list(common)}</section>
<section class="research-slot"><h2>项目与实验</h2><p class="research-status">项目介绍、代码仓库与实验记录待更新。</p></section>
<p><a class="button" href="/index.html#research">返回研究方向</a></p></article></main>
<div id="copyright"><span>Unique AI Lab</span></div></div></body></html>\n'''


def relative_links(source, output, target):
    prefix = os.path.relpath(output, target.parent).replace('\\', '/') + '/'
    updated = re.sub(r'((?:href|src|poster|action|data)=)([\"\'])/(?!/)([^\"\']*)',
                     lambda m: m.group(1) + m.group(2) + prefix + m.group(3), source)
    updated = re.sub(r'url\(/(?!/)([^)]*)\)', lambda m: 'url(' + prefix + m.group(1) + ')', updated)
    return '\n'.join(new.rstrip() if new != old else new
                     for old, new in zip(source.split('\n'), updated.split('\n')))


def pdf_note_page(template, note, directions):
    direction = next(d for d in directions if d['slug'] == note['direction'])
    page = detail(template, direction, [])
    page = re.sub(r'<title>.*?</title>', f'<title>{escape(note["title"])} · Haoran Qian · Unique AI Lab</title>', page)
    categories = ' · '.join(f'<a href="/research/{d["slug"]}/index.html#notes">{escape(d["name"])}</a>' for d in directions if belongs_to(note, d['slug']))
    content = f'''<main id="main"><article class="research-detail pdf-note">
<header><h1>{escape(note['title'])}</h1><p>Haoran Qian · 收录于 <time datetime="{note['date']}">{note['date']}</time> · {note['pages']} 页</p>
<p>{categories}</p></header><p>{escape(note['summary'])}</p>
<div class="pdf-actions"><a class="button" href="/{note['pdf']}" target="_blank" rel="noopener">打开 PDF 阅读</a>
<a class="button" href="/{note['pdf']}" download="{escape(note['title'], quote=True)}.pdf">下载 PDF</a></div>
<object class="pdf-reader" data="/{note['pdf']}" type="application/pdf" aria-label="{escape(note['title'], quote=True)} PDF 原文">
<p>当前浏览器无法嵌入 PDF，请使用上方「打开 PDF 阅读」或「下载 PDF」。</p></object>
<p><a href="/research/{direction['slug']}/index.html#notes">← 返回{escape(direction['name'])}笔记</a></p>
</article></main>'''
    return re.sub(r'<main id="main">.*?</main>', lambda m: content, page, count=1, flags=re.S)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    output = parser.parse_args().output.resolve()
    directions = json.loads((HERE / 'directions.json').read_text(encoding='utf-8'))
    notes = json.loads((HERE / 'notes.json').read_text(encoding='utf-8'))
    template = prepare((HERE / 'legacy-template.html').read_text(encoding='utf-8'), directions)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'assets').mkdir(exist_ok=True)
    shutil.copytree(HERE / 'covers', output / 'assets/research-covers', dirs_exist_ok=True)
    shutil.copytree(HERE / 'pdfs', output / 'assets/notes', dirs_exist_ok=True)
    (output / 'assets/homepage.css').write_text((HERE / 'site.css').read_text(encoding='utf-8'), encoding='utf-8')
    (output / 'index.html').write_text(home(template, directions, notes), encoding='utf-8')
    for direction in directions:
        target = output / 'research' / direction['slug'] / 'index.html'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(detail(template, direction, notes), encoding='utf-8')
    for note in notes:
        if 'pdf' in note:
            target = output / note['path'] / 'index.html'
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(pdf_note_page(template, note, directions), encoding='utf-8')
    for target in output.rglob('*.html'):
        source = target.read_text(encoding='utf-8')
        updated = relative_links(research_navigation(source, directions), output, target)
        if updated != source:
            target.write_text(updated, encoding='utf-8')
    print(f'Built original-theme homepage and {len(directions)} direction pages')


if __name__ == '__main__':
    main()
