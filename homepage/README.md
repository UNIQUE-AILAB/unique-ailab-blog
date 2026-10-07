# 团队主页

主页沿用原网站的 Hexo Mic 主题、封面与导航。原主页文章已按方向收录为技术笔记，主页保留研究方向入口。
方向设置参考 [2026 AI 秋招指北](https://guidebook.hustunique.com/docs/AI%E5%85%A5%E9%97%A8%E6%8C%87%E5%8C%97)。
已有笔记分为计算机视觉、强化学习、自然语言处理三类；AI 基础知识作为各方向共用的通用基础笔记。尚无文章的方向保留空状态。

## 修改内容

- `directions.json`：七个方向的名称、简述、标签及页面路径。
- `notes.json`：原有文章的标题、日期、作者、分类、摘要及原文路径。新增笔记时维护此索引，原文保存在 `source/_posts/`。
- `legacy-template.html`：原网站主页模板，保留原始布局与风格。
- `site.css`：仅新增研究栏目与占位页所需的局部布局样式。
- `build.py`：页面结构与占位区块。

## 生成网站

需要 Python 3.9+，不需要额外依赖。在本仓库根目录运行：

```powershell
python homepage/build.py --output ../UNIQUE-AILAB.github.io
```

输出目标应是网站 fork 的检出目录：
`https://github.com/AetherNoah/UNIQUE-AILAB.github.io`。
脚本生成主页、CSS 和七个方向页，并修正历史 HTML 中的根路径链接以支持项目站点。
重复构建不会重复添加路径前缀，也不会改写历史文章正文。

提交本仓库的源码修改，再提交并推送网站仓库的生成结果。
网站由 GitHub Pages 的 `master` 分支根目录提供服务。

## 本地预览

```powershell
python -m http.server 8765 --directory ../UNIQUE-AILAB.github.io
```

访问 `http://localhost:8765/`。后续填充方向页内容时，修改生成器中的对应区块。

旧 Hexo 源码和历史输出仍保留。不要运行 `hexo g -d` 发布新版主页，
否则旧主题会覆盖新页面；旧部署配置已停用。
