# Contributing

## 首次克隆（必读）

仓库里 GIF、视频、PSD、ZIP 等大文件全部走 Git LFS。**没装 git-lfs 直接 clone，本地看到的图/视频会是几行文本（pointer 文件），不是真内容。**

```bash
# 1. 装 git-lfs（每台机器只需一次）
brew install git-lfs        # macOS
# sudo apt install git-lfs  # Ubuntu / Debian
# choco install git-lfs     # Windows

# 2. clone 之后在仓库目录里
git lfs install   # 给当前仓库挂上 lfs 过滤器
git lfs pull      # 把 LFS 文件真正下载下来

# 3. 验证：随便挑一个视频看大小，几 MB 就对了
ls -lh docs/foundations/simulation/figs/isaac-sim-cube-demo.mp4
```

如果某天发现页面里图/视频坏了，第一反应跑一次 `git lfs pull`。

## 本地预览

本地和 CI 使用 `.nvmrc` 指定的 Node.js 及其自带的 npm。使用 [nvm](https://github.com/nvm-sh/nvm#installing-and-updating) 时，在仓库根目录切换版本；也可通过其他方式安装同一版本：

```bash
nvm install
nvm use
```

### 完整站点预览

完成上面的 Git LFS 初始化与下载后，运行：

```bash
sh local-build.sh
```

脚本会按需安装依赖、构建全部语言并启动预览。打开终端打印的完整地址（含 `/dive-into-embodied-ai/`），同一局域网内的设备也可访问；没有局域网 IPv4 时使用 `localhost`。保持终端运行，按 `Ctrl+C` 停止。

默认端口为 3002，可用 `sh local-build.sh 3003` 指定起始端口。修改页面后再次运行即可重新构建并重启当前仓库的预览；若端口由其他程序占用，脚本会选用后续端口。自动重启需要 `lsof`，macOS 通常自带，Ubuntu/Debian 可用 `sudo apt install lsof` 安装。

### 开发与翻译

日常编辑可启动自动刷新的开发服务器，一次运行一种语言：

```bash
npm run install:locked       # 首次使用或依赖变更后安装
npm run dev                  # 中文，不重新下载 LFS 文件
# 或选择英文 / 意大利语
npm run start -- --locale en
npm run start -- --locale it
```

中文路径为 `/dive-into-embodied-ai/`，英文和意大利语分别使用 `/en/`、`/it/` 子路径。完整构建后可在导航栏切换语言。

翻译章节放在 `i18n/<locale>/docusaurus-plugin-content-docs/current/`，保留源文件目录结构、文档 ID、slug 和显式锚点。界面译文位于 `i18n/<locale>/code.json`，导航和页脚译文位于 `i18n/<locale>/docusaurus-theme-classic/`。其中 `<locale>` 为 `en` 或 `it`；缺少译稿时，页面显示中文原文与翻译范围提示。

发布前检查所有语言：

```bash
npm run build
npm run check:i18n
npm run check:page-markdown
```

### 构建与依赖安装

使用 `npm run build` 构建、`npm run install:locked` 安装依赖；`local-build.sh` 也使用这两个命令。同一工作目录同时只能执行一个构建或依赖安装，进程结束后锁自动释放，无需删除锁文件。直接运行 `npx docusaurus build` 或 `npm ci` 会绕过保护。

`package.json` 和 `.npmrc` 检查 Node/npm 版本；`.npmrc` 也为 npm 脚本关闭 Node 实验性 Web Storage，无需额外设置 `NODE_OPTIONS`。`local-build.sh` 会优先检查当前运行时，版本不符时尝试本机已下载的项目运行时，否则提示切换版本；构建失败时不会启动预览。

## 提交规则

- **位图用 WebP**。不要提交新的 `png` / `jpg` / `jpeg`，先跑 `npm run assets:webp`。
- **大文件走 LFS**。GIF、视频、PSD、ZIP 等不要把真实二进制提交进 Git 历史。
- **招聘岗位数据不要提交进仓库**。后续由外部服务、定时任务或 artifact 发布。
- **提交前至少跑** `npm run assets:webp:check` 和 `npm run build`。

## 本机 SSH 配置（可选）

`git push` 会先触发 `.husky/pre-push` → `git lfs pre-push`，过程中会开多次 SSH 连接。如果没把 key 挂进 ssh-agent，每次都会问 passphrase。macOS 一次性解决：

```bash
ssh-add --apple-use-keychain ~/.ssh/id_rsa
```

并在 `~/.ssh/config` 里给 `Host github.com` 加上：

```
Host github.com
  AddKeysToAgent yes
  UseKeychain yes
  IdentityFile ~/.ssh/id_rsa
```

之后 push 就不会反复问 passphrase 了，重启后也会自动从 Keychain 恢复。
