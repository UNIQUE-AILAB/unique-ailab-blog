"""Add research placeholders to the original website, preserving its theme."""
import argparse
import html
import json
import os
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = 'https://github.com/UNIQUE-AILAB/UNIQUE-AILAB.github.io'
escape = html.escape


def belongs_to(note, slug):
    return note['direction'] == slug or slug in note.get('related_directions', [])


def research_navigation(source, directions):
    if 'assets/homepage.css' not in source:
        source = source.replace('</head>', '<link rel="stylesheet" href="/assets/homepage.css">\n</head>', 1)
    entries = ''.join(f'<li><a class="category-link" href="/research/{d["slug"]}/index.html">{escape(d["name"])}</a></li>' for d in directions)
    project_active = re.search(r'<li class="active">\s*<a href="[^\"]*project/', source)
    menu_class = 'research-navigation' if project_active else 'active research-navigation'
    menu = f'<li class="{menu_class}"><a href="/index.html#research">研究方向</a><ul class="submenu">{entries}</ul></li>'
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
        items.append(f'''<li class="research-note"><h3><a href="/{escape(note['path'], quote=True)}">{escape(note['title'])}</a></h3>
<p class="research-status">{date_label} <time datetime="{note['date']}">{note['date']}</time>{author}</p>
<p>{escape(note['summary'])}</p><a href="/{escape(note['path'], quote=True)}">阅读全文 →</a></li>''')
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


def project_shell(template, directions, title, content):
    page = detail(template, directions[0], [])
    page = re.sub(r'<title>.*?</title>', f'<title>{escape(title)} · Unique AI Lab</title>', page)
    page = page.replace('class="active research-navigation"', 'class="research-navigation"')
    page = re.sub(r'<li>\s*(<a href="/project/"[^>]*>)', r'<li class="active">\1', page)
    return re.sub(r'<main id="main">.*?</main>', lambda m: f'<main id="main">{content}</main>', page, count=1, flags=re.S)


def project_tags(project):
    return '<ul class="project-tags" aria-label="项目技术与方向">' + ''.join(
        f'<li>{escape(tag)}</li>' for tag in project['tags']) + '</ul>'


def project_index(template, directions, projects):
    cards = []
    for project in projects:
        path = f'/project/{project["slug"]}/index.html'
        cards.append(f'''<article class="project-card">
<a class="project-card-image" href="{path}" tabindex="-1" aria-hidden="true"><img src="/assets/project-images/{escape(project['cover'])}" alt="" width="1440" height="1000" loading="lazy"></a>
<div class="project-card-content"><p class="project-award">{escape(project['award'])}</p>
<h2><a href="{path}">{escape(project['name'])}</a></h2><p class="project-subtitle">{escape(project['subtitle'])}</p>
<p>{escape(project['summary'])}</p>{project_tags(project)}
<div class="project-actions"><a class="button special" href="{path}">了解项目</a><a class="button" href="{escape(project['repository'], quote=True)}" target="_blank" rel="noopener noreferrer">GitHub 仓库 ↗</a></div></div></article>''')
    content = f'''<section class="research-detail project-page"><header><p class="project-eyebrow">UNIQUE AI LAB</p><h1>项目</h1><p>团队项目、竞赛作品与工程实践。</p></header>
<div class="project-list">{''.join(cards)}</div></section>'''
    return project_shell(template, directions, '项目', content)


def project_detail(template, directions, project):
    features = ''.join(f'<section><h3>{escape(feature["title"])}</h3><p>{escape(feature["description"])}</p></section>' for feature in project['features'])
    content = f'''<article class="research-detail project-page project-detail">
<p class="project-back"><a href="/project/index.html">← 全部项目</a></p>
<header><p class="project-eyebrow">PROJECT / {escape(project['slug'].upper())}</p><h1>{escape(project['name'])}</h1>
<p class="project-subtitle">{escape(project['subtitle'])}</p><p class="project-award">{escape(project['award'])}</p>
<div class="project-actions"><a class="button special" href="{escape(project['repository'], quote=True)}" target="_blank" rel="noopener noreferrer">查看 GitHub 仓库 ↗</a><a class="button" href="{escape(project['repository'], quote=True)}/#快速开始" target="_blank" rel="noopener noreferrer">运行与文档 ↗</a></div></header>
<figure class="project-screenshot"><img src="/assets/project-images/{escape(project['cover'])}" alt="FraudLens 产品首页，包含风险研判、知识库、举报材料与情景模拟入口" width="1440" height="1000"><figcaption>产品首页 · 本地离线演示</figcaption></figure>
<section class="research-slot"><h2>项目介绍</h2><p>{escape(project['summary'])}</p>{project_tags(project)}
<div class="project-features">{features}</div></section>
<section class="research-slot"><h2>设计思路</h2><p>诈骗线索会随着对话不断变化。FraudLens 将风险判断、命中依据和下一步行动放在同一条交互链路中：持续理解用户补充的信息，用知识与证据解释风险，再通过报告与互动学习承接后续操作。</p>
<ol class="project-pipeline"><li>接收可疑内容，追踪会话事实与风险阶段</li><li>检索本地知识，完成规则型多 Agent 分析与交叉核验</li><li>展示分级劝阻、证据解释与场景化报告</li><li>整理本地举报草稿，通过情景模拟完成复盘</li></ol>
<p>核心链路采用本地优先设计，完成环境准备后可以离线演示；联网时可选用外部模型改善最终回复表达，风险等级仍由本地链路产生。</p></section>
<section class="research-slot"><h2>真实研判界面</h2><figure class="project-screenshot"><img src="/assets/project-images/{escape(project['analysis_image'])}" alt="FraudLens 对虚构刷单案例给出的离线风险研判、劝阻回复与知识证据" width="1440" height="1000" loading="lazy"><figcaption>使用虚构案例，展示实际后端返回的研判与证据</figcaption></figure></section>
<section class="research-slot"><h2>成果与代码</h2><p class="project-award">{escape(project['award'])}</p><p>代码、安装说明、系统架构与开发文档见 <a href="{escape(project['repository'], quote=True)}" target="_blank" rel="noopener noreferrer">Arison591/fraudlens ↗</a>。仓库 fork 自 <a href="{escape(project['upstream'], quote=True)}" target="_blank" rel="noopener noreferrer">chenge-skr/raicom- ↗</a>，保留原始提交历史。</p>
<p class="research-status">当前版本用于反诈教育与项目演示。举报功能保存本地草稿，运行要求与功能范围详见仓库说明。</p></section>
</article>'''
    return project_shell(template, directions, project['name'], content)


def article_page(template, note, directions):
    direction = next(d for d in directions if d['slug'] == note['direction'])
    page = detail(template, direction, [])
    page = re.sub(r'<title>.*?</title>', f'<title>{escape(note["title"])} · {escape(note["author"])} · Unique AI Lab</title>', page)
    page = page.replace('</head>', '<link rel="stylesheet" href="/css/typo.css">\n</head>')
    categories = ' · '.join(f'<a href="/research/{d["slug"]}/index.html#notes">{escape(d["name"])}</a>' for d in directions if belongs_to(note, d['slug']))
    body = (HERE / note['content']).read_text(encoding='utf-8')
    content = f'''<main id="main"><article class="research-detail note-article">
<header><h1>{escape(note['title'])}</h1><p>{escape(note['author'])} · 收录于 <time datetime="{note['date']}">{note['date']}</time></p>
<p>{categories}</p></header>
<div class="typo note-body">{body}</div>
<p><a href="/research/{direction['slug']}/index.html#notes">← 返回{escape(direction['name'])}笔记</a></p>
</article></main>'''
    return re.sub(r'<main id="main">.*?</main>', lambda m: content, page, count=1, flags=re.S)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    output = parser.parse_args().output.resolve()
    directions = json.loads((HERE / 'directions.json').read_text(encoding='utf-8'))
    notes = json.loads((HERE / 'notes.json').read_text(encoding='utf-8'))
    projects = json.loads((HERE / 'projects.json').read_text(encoding='utf-8'))
    template = prepare((HERE / 'legacy-template.html').read_text(encoding='utf-8'), directions)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'assets').mkdir(exist_ok=True)
    shutil.copytree(HERE / 'covers', output / 'assets/research-covers', dirs_exist_ok=True)
    shutil.copytree(HERE / 'note-images', output / 'assets/note-images', dirs_exist_ok=True)
    shutil.copytree(HERE / 'project-images', output / 'assets/project-images', dirs_exist_ok=True)
    (output / 'assets/homepage.css').write_text((HERE / 'site.css').read_text(encoding='utf-8'), encoding='utf-8')
    (output / 'index.html').write_text(home(template, directions, notes), encoding='utf-8')
    project_root = output / 'project'
    project_root.mkdir(exist_ok=True)
    (project_root / 'index.html').write_text(project_index(template, directions, projects), encoding='utf-8')
    for project in projects:
        target = project_root / project['slug'] / 'index.html'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(project_detail(template, directions, project), encoding='utf-8')
    for direction in directions:
        target = output / 'research' / direction['slug'] / 'index.html'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(detail(template, direction, notes), encoding='utf-8')
    for note in notes:
        if 'content' in note:
            target = output / note['path'] / 'index.html'
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(article_page(template, note, directions), encoding='utf-8')
    for target in output.rglob('*.html'):
        source = target.read_text(encoding='utf-8')
        updated = relative_links(research_navigation(source, directions), output, target)
        if updated != source:
            target.write_text(updated, encoding='utf-8')
    print(f'Built homepage, {len(directions)} direction pages and {len(projects)} project pages')


if __name__ == '__main__':
    main()
