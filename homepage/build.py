"""Add research placeholders to the original website, preserving its theme."""
import argparse
import html
import json
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
GUIDE = 'https://guidebook.hustunique.com/docs/AI%E5%85%A5%E9%97%A8%E6%8C%87%E5%8C%97'
REPO = 'https://github.com/AetherNoah/UNIQUE-AILAB.github.io'
escape = html.escape


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


def home(template, directions):
    cards = []
    for direction in directions:
        cards.append(f'''<article class="link_box special research-card">
<h3><a href="/research/{direction['slug']}/index.html">{escape(direction['name'])}</a></h3>
<p class="research-english">{escape(direction['english'])}</p>
<p>{escape(direction['summary'])}</p>
<a class="button" href="/research/{direction['slug']}/index.html">查看方向详情</a></article>''')
    section = f'''<section id="research" class="research-area" aria-labelledby="research-title">
<header class="link_box special research-heading"><h2 id="research-title">研究方向</h2>
<p>了解各方向的研究主题、基础知识与学习资料。</p>
<p class="research-source">学习指引：<a href="{GUIDE}">AI 入门指北</a></p></header>
<div class="research-grid">{''.join(cards)}</div></section>\n'''
    marker = '<!-- 文章列表   s -->'
    assert marker in template
    return template.replace(marker, section + marker, 1)


def detail(template, direction, guide):
    head = template.split('</head>', 1)[0] + '</head>'
    head = re.sub(r'<title>.*?</title>', f'<title>{escape(direction["name"])} · Unique AI Lab</title>', head, flags=re.S)
    nav = template.split('<nav id="nav"', 1)[1].split('</nav>', 1)[0]
    nav = '<nav id="nav"' + nav + '</nav>'
    slots = [('项目与实验', '项目介绍、代码仓库与实验记录待更新。'),
             ('论文与笔记', '论文阅读、组会分享与复现笔记待更新。'),
             ]
    content = ''.join(f'<section class="research-slot"><h2>{title}</h2><p>{text}</p><p class="research-status">暂无内容 · 待更新</p></section>' for title, text in slots)
    topics = ''.join(f'<li><strong>{escape(item["title"])}</strong>：{escape(item["text"])}</li>' for item in guide['sections'])
    resources = ''.join(f'<li><a href="{escape(item["url"], quote=True)}">{escape(item["name"])}</a></li>' for item in guide['resources'])
    return f'''{head}<body class="is-loading"><div id="wrapper" class="fade-in">
<header id="header"><a href="/index.html" class="logo">UNIQUE AI</a></header>{nav}
<main id="main"><article class="research-detail"><header><h1>{escape(direction['name'])}</h1>
<p>{escape(direction['english'])}</p></header><p>{escape(direction['summary'])}</p>
<section class="research-slot" id="topics"><h2>细分主题</h2><ul>{topics}</ul></section>
<section class="research-slot" id="getting-started"><h2>入门与基础</h2><p>{escape(guide['foundations'])}</p></section>
<section class="research-slot" id="resources"><h2>学习资料</h2><ul>{resources}</ul><p class="research-source">本页学习内容整理自 <a href="{GUIDE}">AI 入门指北</a>。</p></section>
{content}<p><a class="button" href="/index.html#research">返回研究方向</a></p></article></main>
<div id="copyright"><span>Unique AI Lab</span></div></div></body></html>\n'''


def relative_links(source, output, target):
    prefix = os.path.relpath(output, target.parent).replace('\\', '/') + '/'
    updated = re.sub(r'((?:href|src|poster|action)=)([\"\'])/(?!/)([^\"\']*)',
                     lambda m: m.group(1) + m.group(2) + prefix + m.group(3), source)
    updated = re.sub(r'url\(/(?!/)([^)]*)\)', lambda m: 'url(' + prefix + m.group(1) + ')', updated)
    return '\n'.join(new.rstrip() if new != old else new
                     for old, new in zip(source.split('\n'), updated.split('\n')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    output = parser.parse_args().output.resolve()
    directions = json.loads((HERE / 'directions.json').read_text(encoding='utf-8'))
    guides = json.loads((HERE / 'guide.json').read_text(encoding='utf-8'))
    template = prepare((HERE / 'legacy-template.html').read_text(encoding='utf-8'), directions)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'assets').mkdir(exist_ok=True)
    (output / 'assets/homepage.css').write_text((HERE / 'site.css').read_text(encoding='utf-8'), encoding='utf-8')
    (output / 'index.html').write_text(home(template, directions), encoding='utf-8')
    for direction in directions:
        target = output / 'research' / direction['slug'] / 'index.html'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(detail(template, direction, guides[direction['slug']]), encoding='utf-8')
    for target in output.rglob('*.html'):
        source = target.read_text(encoding='utf-8')
        updated = relative_links(research_navigation(source, directions), output, target)
        if updated != source:
            target.write_text(updated, encoding='utf-8')
    print(f'Built original-theme homepage and {len(directions)} direction pages')


if __name__ == '__main__':
    main()
