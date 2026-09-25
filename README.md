# Transformer Atlas

A public, frontend-only simulation of the original Transformer architecture.

## Use

Scroll to learn. Open **Your text** to supply source and expected output text
(up to six tokens per side). The starting copy task is **rain feeds rivers**.
Use the step slider to inspect calculations, choose 2D or 3D, and inspect spatial
scenes from Front, Top or Perspective. All weights start untrained.

## Hosting

This repository is ready for GitHub Pages: **Settings → Pages → Deploy from a
branch → main → / (root)**. It contains no API keys or server dependencies.
Every JS and CSS file in the root is editable frontend source.

## Complete development project

Download `transformer-atlas-source.zip` for the organized development project,
validation suite, local preview server, hosting guide, and optional GitHub Actions
workflow. That workflow deploys the development project's `build/` directory.

## Educational scope

70 reference diagram concepts are reconstructed with frontend assets. Native
MathML typesets formulas; WebGL renders actual meshes. The browser model has
four features, two heads, and eight feed-forward units. It performs real forward
arithmetic, reverse-mode gradients and clipped SGD. This is an educational
miniature, not the paper's trained model or a general translator.

Architecture attribution: Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

## Equation rendering checks

Run `node --test math.test.mjs` (Node.js 20 or later). The checks cover fraction grouping, Unicode scripts, escaped text, mathematical display layout, and the formulas and animation labels across all 394 chapters. Equations use the bundled OpenType MATH font and native MathML, with no external typesetting service.
