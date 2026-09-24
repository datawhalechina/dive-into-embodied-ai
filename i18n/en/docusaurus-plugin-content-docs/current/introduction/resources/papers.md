---
title: Papers and research
description: Papers, project pages, code, and explanations organized by research topic.
sidebar_position: 10
displayed_sidebar: introductionSidebar
---

# Papers and research

Start from a research question and follow links to the original paper, the authors' project page, and explanations on this site. The first entries cover work related to existing tutorials; topics such as locomotion, perception, and world models will be added over time.

## Imitation learning and action generation \{#imitation-learning}

### ACT · Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware \{#act}

**2023 · Bimanual manipulation / imitation learning / action chunking**

Studies how low-cost bimanual hardware can learn fine-grained manipulation from demonstrations. ACT predicts a sequence of actions at once and is designed around compounding errors in long-horizon imitation learning.

- Original sources: [paper](https://arxiv.org/abs/2304.13705) · [authors' project page](https://tonyzhaozh.github.io/aloha/)
- Read on this site: [ACT architecture and key mechanisms](/docs/foundations/vla/act_action_chunking)
- Try it: [ACT bimanual training](/docs/practices/vla/act)

### Diffusion Policy · Visuomotor Policy Learning via Action Diffusion \{#diffusion-policy}

**2023 · Visuomotor policies / diffusion models / continuous actions**

Models robot action generation as a conditional denoising process to represent multimodal action distributions, and combines visual conditioning with receding-horizon execution to generate action sequences.

- Original sources: [paper](https://arxiv.org/abs/2303.04137) · [authors' project page](https://diffusion-policy.cs.columbia.edu/)
- Read on this site: [Diffusion Policy action modeling](/docs/foundations/vla/diffusion_policy)

## Vision-language-action models \{#vla}

### OpenVLA · An Open-Source Vision-Language-Action Model \{#openvla}

**2024 · VLA / multi-robot data / fine-tuning**

Connects a vision-language model to robot action prediction and releases the model and training code, making it easier to study cross-task learning and fine-tuning for new robot settings.

- Original sources: [paper](https://arxiv.org/abs/2406.09246) · [authors' project page](https://openvla.github.io/)
- Read on this site: [OpenVLA reproduction and key trade-offs](/docs/foundations/vla/openvla_engineering)
- Related resources: [Embodied AI datasets](./datasets) · [Open source projects](./open-source)

Compiled and sources checked: 2026-09-11.
