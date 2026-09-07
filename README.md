# ST-Mem

Project page for **Linguistic Trajectory Encoding for Efficient Long-Horizon
Spatial Memory in Embodied Agents**.

[Project page](https://sealical.github.io/st-mem/) ·
[Paper](https://arxiv.org/abs/2609.04802) · [BibTeX](citation.bib)

**Code coming soon.** This repository contains the public website only, not the
ST-Mem implementation, model weights, training data, or evaluation pipelines.
The page describes the planned core release; its command example is a preview
for when that release becomes available. The paper is currently presented as
an arXiv preprint, without a conference acceptance claim.

## Preview locally

From this repository's root:

```bash
python -m http.server 8765 --bind 127.0.0.1
```

Open <http://127.0.0.1:8765/>. No build step, backend, Node packages, or GPU is
required. The website uses plain HTML, CSS, and JavaScript, with no external
fonts, scripts, analytics, or third-party website template.

## Deployment and updates

GitHub Pages publishes the `main` branch's root directory at
<https://sealical.github.io/st-mem/>. The `.nojekyll` file keeps the site static.
Push website changes to `main` to trigger an automatic redeployment; check the
repository's **Actions** tab and **Settings → Pages** for the deployment result.

Keep `index.html`, `assets/`, `citation.bib`, `.nojekyll`, and `LICENSE` at the
published root. If the public URL changes, update the canonical, Open Graph,
and Twitter image metadata in `index.html` together. When the implementation
is publicly released, replace the “Code coming soon” status with its verified
public repository and installation links.

Paper figures open at their original resolution. Task tabs are illustrative
examples, not live inference. Results are attributed to the paper, not claimed
as independently reproduced by this website.

## Citation

Please cite the paper rather than the website:

```bibtex
@misc{xie2026linguistictrajectory,
  title = {Linguistic Trajectory Encoding for Efficient Long-Horizon Spatial Memory in Embodied Agents},
  author = {Tianyidan Xie and Shenyi Wang and Qiang Tang and Mingjie Wang and Zhicheng Qiu and Xuanfu Li and Zhan Xu and Jian Yang and Lanjun Wang and Zili Yi},
  year = {2026},
  eprint = {2609.04802},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  doi = {10.48550/arXiv.2609.04802},
  url = {https://arxiv.org/abs/2609.04802}
}
```

## License and figure attribution

Original website code is available under the [MIT License](LICENSE).
Figures 1 and 2 are unmodified author-supplied images from the paper, under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), not MIT. Preserve the
[figure attribution](assets/README.md) when redistributing them.
