# Tabular Generator Candidates by Priority

Last updated: 2026-05-01

This list is prioritized for the current PRELIM codebase, which already includes statistical baselines, KDE/GMM variants, SMOTE/ADASYN, CTGAN, TVAE, CopulaGAN, GaussianCopula, TabGAN, ForestDiffusion, and a Bayesian-network wrapper.

Priority reflects expected benchmark value, implementation usefulness, and whether the model adds a genuinely new family rather than duplicating what is already covered.

## 1. TabSyn

- Why prioritize it:
  - Strong modern tabular synthesis baseline.
  - Latent-space diffusion makes it meaningfully different from the current wrappers.
  - Good candidate if only one additional recent neural model is added next.
- Paper:
  - ICLR 2024 oral: https://arxiv.org/abs/2310.09656
- Official implementation:
  - https://github.com/amazon-science/tabsyn

## 2. TabDDPM

- Why prioritize it:
  - Canonical feature-space diffusion baseline for mixed-type tabular data.
  - Widely cited and still a useful comparison point in newer work.
  - Complements `ForestDiffusion` rather than duplicating it exactly.
- Paper:
  - ICML 2023: https://research.yandex.com/publications/tab-ddpm-modelling-tabular-data-with-diffusion-models
  - arXiv: https://arxiv.org/abs/2209.15421
- Official implementation:
  - https://github.com/yandex-research/tab-ddpm

## 3. GReaT

- Why prioritize it:
  - Adds an autoregressive LLM-style tabular generator family.
  - More mature and package-like than many research-only repos.
  - Useful if PRELIM should compare classic tabular models against language-model-based synthesis.
- Paper:
  - "Language Models are Realistic Tabular Data Generators": https://arxiv.org/abs/2210.06280
- Implementation:
  - https://github.com/tabularis-ai/be_great

## 4. TabuLa

- Why prioritize it:
  - Newer LLM-style generator than GReaT.
  - Focuses on tabular-specific training rather than relying on pretrained NLP weights in the same way.
  - Good follow-up if an LLM family is added and a second, more recent variant is desired.
- Paper:
  - PAKDD 2025 record: https://dblp.org/rec/conf/pakdd/ZhaoBC25.html
  - arXiv record: https://dblp.org/rec/journals/corr/abs-2310-12746.html
- Official implementation:
  - https://github.com/zhao-zilong/Tabula

## 5. STaSy

- Why prioritize it:
  - Important early score-based tabular synthesis reference.
  - Still useful historically and as an additional diffusion-family baseline.
  - Lower priority than TabSyn and TabDDPM because those are the stronger first additions today.
- Paper:
  - ICLR 2023 / arXiv: https://arxiv.org/abs/2210.04018
- Official implementation:
  - https://github.com/JayoungKim408/STaSy

## 6. CTAB-GAN+

- Why prioritize it:
  - Solid historical GAN baseline with an official implementation.
  - Still worth including if the benchmark wants broader GAN-family coverage.
  - Lower priority because PRELIM already has several GAN/copula-style generators, so the marginal coverage gain is smaller.
- Paper:
  - Paper page: https://arxiv.org/abs/2204.00401
  - Journal version: https://pmc.ncbi.nlm.nih.gov/articles/PMC10801038/
- Official implementation:
  - https://github.com/Team-TUD/CTAB-GAN-Plus

## 7. Tabby

- Why prioritize it:
  - Interesting recent LLM architecture modification specifically for tabular synthesis.
  - Potentially high upside, but less established than GReaT or TabuLa.
  - Best treated as an experimental recent addition rather than an early integration target.
- Paper:
  - arXiv: https://arxiv.org/abs/2503.02152
- Official implementation:
  - https://github.com/soCromp/tabby

## 8. Binary Diffusion

- Why prioritize it:
  - Compact and interesting recent diffusion variant using binary representations.
  - Useful as an experimental addition if PRELIM expands its diffusion coverage further.
  - Lower priority than TabDDPM and TabSyn because it is less established as a default baseline.
- Paper:
  - arXiv: https://arxiv.org/abs/2409.13882
- Official implementation:
  - https://github.com/vkinakh/binary-diffusion-tabular

## Suggested Integration Order

1. TabSyn
2. TabDDPM
3. GReaT
4. TabuLa
5. STaSy
6. CTAB-GAN+
7. Tabby
8. Binary Diffusion

## Notes

- This ranking is practical, not absolute. If the goal is specifically "cover every major model family", then `GReaT` moves up because it is the clearest LLM/autoregressive representative.
- If the goal is specifically "recent diffusion models only", then `STaSy` and `Binary Diffusion` move up relative to the LLM entries.
- Some newer papers exist, but this list prefers methods with accessible public implementations over paper-only candidates.
