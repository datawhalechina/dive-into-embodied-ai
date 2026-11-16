"""SmolVLM2 骨干下载（SmolVLA 0.6.1 的 VLM 按仓库 id 单独加载，smolvla_base 快照不含它）。"""
from huggingface_hub import snapshot_download

p = snapshot_download("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
print("VLM_OK", p)
