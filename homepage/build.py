"""Build the standalone team homepage without the legacy Hexo theme.

Usage: python homepage/build.py --output ../UNIQUE-AILAB.github.io
"""
import argparse
import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
GUIDE = "https://guidebook.hustunique.com/docs/AI%E5%85%A5%E9%97%A8%E6%8C%87%E5%8C%97"
REPO = "https://github.com/AetherNoah/UNIQUE-AILAB.github.io"
escape = html.escape


def page(title, body, prefix="./"):
    return f'''<!doctype html>
<html lang="zh-CN">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)} · UNIQUE AI LAB</title>
<meta name="description" content="联创 AI 组的研究方向、项目与学习记录。">
<link rel="icon" href="{prefix}img/favicon.png">
<link rel="stylesheet" href="{prefix}assets/homepage.css"></head>
<body><a class="skip" href="#main">跳转到正文</a><div class="wrap">
<header class="header"><a class="brand" href="{prefix}index.html">UNIQUE <span>AI LAB</span></a>
<nav class="nav" aria-label="主导航"><a href="{prefix}index.html#research">研究方向</a><a href="{GUIDE}">入门指北 ↗</a><a href="{REPO}">GitHub ↗</a></nav></header>
<main id="main">{body}</main>
<footer class="footer"><span>UNIQUE AI LAB · 联众人之志，创非凡之事</span><span>基于 <a class="text-link" href="https://github.com/UNIQUE-AILAB">UNIQUE-AILAB</a> 历史网站延续维护</span></footer>
</div></body></html>\n'''


def tags(direction):
    return '<ul class="topics">' + ''.join(f'<li>{escape(topic)}</li>' for topic in direction['topics']) + '</ul>'


def home(directions):
    cards = []
    for i, direction in enumerate(directions, 1):
        cards.append(f'''<a class="card" href="./research/{direction['slug']}/index.html" aria-label="查看{escape(direction['name'])}栏目">
<div class="card-top"><span class="code">{i:02d} / {escape(direction['code'])}</span><span class="status">内容筹备中</span></div>
<h3>{escape(direction['name'])}</h3><div class="english">{escape(direction['english'])}</div>
<p>{escape(direction['summary'])}</p>{tags(direction)}<span class="card-link">进入方向栏目 <span aria-hidden="true">↗</span></span></a>''')
    diagram = ''.join(f'<span>{escape(d["code"])}</span>' for d in directions)
    return page('首页', f'''<section class="hero" aria-labelledby="hero-title"><div>
<div class="eyebrow">UNIQUE STUDIO / ARTIFICIAL INTELLIGENCE</div>
<h1 id="hero-title">保持好奇，<br>探索<span>智能的边界。</span></h1>
<p class="lead">联创 AI 组 · 聚焦算法、模型与训练方法。<br>在阅读、实验与协作中，把想法变成可以验证的探索。</p>
<div class="actions"><a class="button primary" href="#research">探索研究方向 ↓</a><a class="button" href="{GUIDE}">阅读入门指北 ↗</a></div></div>
<aside class="diagram" aria-label="七个研究方向示意"><div class="diagram-label">RESEARCH MAP / 2026</div><div class="diagram-center">THINK AI.<small>CULTIVATE ELITE.</small></div><div class="diagram-grid">{diagram}</div></aside></section>
<section class="section" id="research" aria-labelledby="research-title"><div class="section-heading"><div><div class="eyebrow">01 / RESEARCH AREAS</div><h2 id="research-title">我们探索的方向</h2></div><p>七个方向，持续积累。<br>项目、论文笔记与学习资料将陆续更新。</p></div>
<div class="grid">{''.join(cards)}</div><p class="note">方向设置参考 <a class="text-link" href="{GUIDE}">2026 AI 秋招指北</a>；各栏目目前为占位，尚未发布项目与成果。</p></section>
<section class="section" aria-labelledby="resources-title"><div class="section-heading"><div><div class="eyebrow">02 / KEEP EXPLORING</div><h2 id="resources-title">从这里继续</h2></div></div>
<div class="resource-grid"><article class="resource"><h3>学习与准备</h3><p>从基础知识出发，找到感兴趣的细分方向，在动手中建立理解。</p><a class="text-link" href="{GUIDE}">前往 AI 入门指北 ↗</a></article>
<article class="resource"><h3>历史文章</h3><p>回看往届成员留下的技术文章与学习记录，继续积累新的探索。</p><a class="text-link" href="./archives/index.html">浏览历史归档 →</a></article></div></section>''')


def detail(direction):
    slots = [('PROJECTS', '项目与实验', '这里将收录本方向的项目介绍、代码仓库与实验记录。'),
             ('READING', '论文与笔记', '这里将收录论文阅读、组会分享与复现笔记。'),
             ('RESOURCES', '学习资料', '这里将收录入门路径、课程与精选学习资源。')]
    empty = ''.join(f'<article class="empty"><div class="eyebrow">{code}</div><h3>{title}</h3><p>{text}</p><span class="empty-label">待更新 · 暂无内容</span></article>' for code, title, text in slots)
    return page(direction['name'], f'''<section class="detail-hero"><a class="back" href="../../index.html#research">← 全部研究方向</a>
<p class="eyebrow">{escape(direction['code'])} / {escape(direction['english'])}</p><h1>{escape(direction['name'])}</h1><p class="lead">{escape(direction['summary'])}</p>
<div class="detail-topics">{tags(direction)}</div></section>
<section class="section" aria-labelledby="content-title"><div class="section-heading"><div><div class="eyebrow">EXPLORE & BUILD</div><h2 id="content-title">方向内容</h2></div><span class="status">内容筹备中</span></div>
<div class="empty-grid">{empty}</div><p class="note">本页为方向占位页。具体研究主题与内容将随团队工作持续更新。</p>
<div class="actions"><a class="button" href="{GUIDE}">查看入门指北 ↗</a><a class="button" href="../../index.html#research">返回研究方向</a></div></section>''', '../../')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    directions = json.loads((HERE / 'directions.json').read_text(encoding='utf-8'))
    output.mkdir(parents=True, exist_ok=True)
    (output / 'assets').mkdir(exist_ok=True)
    (output / 'assets/homepage.css').write_text((HERE / 'site.css').read_text(encoding='utf-8'), encoding='utf-8')
    (output / 'index.html').write_text(home(directions), encoding='utf-8')
    for direction in directions:
        target = output / 'research' / direction['slug']
        target.mkdir(parents=True, exist_ok=True)
        (target / 'index.html').write_text(detail(direction), encoding='utf-8')
    # Historical Hexo output used root-relative URLs; forks are project sites.
    # Resolve those links relative to each page without touching external URLs.
    import os
    import re
    for target in output.rglob('*.html'):
        source = target.read_text(encoding='utf-8')
        def relative_link(match):
            path = match.group(3)
            # Preserve trailing slash, query strings and fragments.
            relative = os.path.relpath(output, target.parent).replace('\\', '/')
            return match.group(1) + match.group(2) + relative + '/' + path
        updated = re.sub(r'((?:href|src|poster|action)=)([\"\'])/(?!/)([^\"\']*)', relative_link, source)
        # Also preserve legacy thumbnails declared inline in HTML.
        updated = re.sub(r'url\(/(?!/)([^)]*)\)', lambda m: 'url(' + os.path.relpath(output, target.parent).replace('\\', '/') + '/' + m.group(1) + ')', updated)
        if updated != source:
            updated = '\n'.join(new.rstrip() if new != old else new
                                for old, new in zip(source.split('\n'), updated.split('\n')))
            target.write_text(updated, encoding='utf-8')
    print(f'Built homepage and {len(directions)} direction pages in {output}')


if __name__ == '__main__':
    main()
