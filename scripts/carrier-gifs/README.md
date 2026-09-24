# 载体演示素材

网页支持“官方演示 / 3D 示意”切换，默认展示官方演示。官方片段使用静音循环 MP4 播放，3D 示意使用 GIF 播放。未进入视野或用户选择停止播放时，仅显示封面。

## 官方演示

原片、官方来源页、截取起止时间与 SHA-256 记录在 `official-sources.json`。六类载体分别选用 Unitree G1、ALOHA 2、Husky A300、Unitree Go2、Mobile ALOHA，以及 ALOHA 2 的 MuJoCo 仿真。

安装 `ffmpeg` 和 Python 的 `Pillow` 后，在仓库根目录运行：

```sh
node scripts/carrier-gifs/encode-official.mjs
```

脚本下载清单中的公开原片到忽略目录 `.agents/carrier-gifs/sources/`，校验内容后生成用于网页播放的 640 × 400、24 FPS MP4 和 WebP 封面，输出到 `static/img/introduction/carriers/official/`。通过等比缩放和补边保留完整画面，不改变原片速度，不裁去水印或速度标注。ALOHA 2 真机片段是原片中标注为 4× 的遥操作演示；仿真片段同样来自官方项目页，不能标为真机演示。

每张卡片链接到对应官方来源，正文与英文翻译在 `src/components/docs/introduction/embodiments/CarrierDemos/content.ts` 和 `i18n/en/code.json` 中维护。第三方素材的权利归原作者；本项目许可证不用于重新授权这些原片。

## 3D 示意

使用项目已有的 Three.js 绘制通用机器人结构，导出为 GIF。模型与动作是教学示意，不对应表格中的具体厂商产品，也不是物理控制策略。

1. 安装项目依赖，并确保本机有 `ffmpeg`。
2. 在仓库根目录运行 `node scripts/carrier-gifs/serve.mjs`。
3. 浏览器打开终端中的本地地址，检查六类载体；点击“生成全部 GIF 帧”。
4. 等待完成后运行 `node scripts/carrier-gifs/encode.mjs`。

每段动画为 640 × 400、20 FPS、4 秒。GIF 与 WebP 静态封面输出到 `static/img/introduction/carriers/`。中间帧保存在忽略目录 `.agents/carrier-gifs/frames/`，不会进入仓库。

`scene.mjs` 管理模型与循环动作，`render.mjs` 逐帧导出，`serve.mjs` 仅在本机回环地址提供渲染工具与帧写入接口。网站本身使用静态 GIF，不运行这一套渲染工具。
