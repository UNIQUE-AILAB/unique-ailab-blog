# 安装
确保你安装了 npm, curl, git
``` shell
./setup.sh
npm install -g hexo-cli
```

# 使用方法

## New post
`hexo new post <title>`


** 要使用斜体时请使用\*标记，而不要用 \_ **

## 文章头部说明
categories: 目录，字符串不需要使用“或'
mathjax: 如要开启数学公式渲染，请设为true
thumbnail: 文章封面图，如果没有会有点丑。建议填http链接，也可本地文件路径。

## 部署
先与远程仓库同步, pull & push
`hexo g -d` or `hexo d -g`
