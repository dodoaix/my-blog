<div class="projects-motto" aria-label="Learning motto">
  <div class="projects-motto__label">MOTTO</div>
  <div id="projects-motto-text" class="projects-motto__text"></div>
</div>

<script>
(() => {
  const mottos = [
    "Don't import it. Build it.",
    "Make hands dirty",
    "what i cannot create i do not understand",
  ];
  const target = document.getElementById("projects-motto-text");

  if (target && mottos.length) {
    target.textContent = mottos[Math.floor(Math.random() * mottos.length)];
  }
})();
</script>

# Projects

这里整理相对完整、可持续更新的算法与工程项目。

## TITO Transformer

TITO Transformer（对应 TinyTorch-Language Focus）参照 TinyTorch 的学习路线，从 Tensor、Module 等底层基本单元出发，逐步搭建面向 GPT 的深度学习训练框架。项目重点是亲手重现数据流、前向传播、反向传播、优化器，以及 Tokenizer、Embedding 和 Attention 等 Transformer 所需组件。

[进入项目 →](tito-transformer/index.md)

[cs249r - 机器学习系统官方仓库](https://github.com/harvard-edge/cs249r_book)

[TinyTorch 项目页](https://mlsysbook.ai/tinytorch/getting-started.html)

*特别鸣谢：Datawhale 14 天学习计划（by 静静）*

## Diffusion from Scratch: From DiT to Image Editing

从 Diffusion 与 DiT 的底层机制出发，逐步完成生成、图像编辑、性能优化与部署的完整实践。

[进入项目 →](diffusion-from-scratch/index.md)
