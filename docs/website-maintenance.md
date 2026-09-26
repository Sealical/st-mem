# Website maintenance

This document is for maintainers of the **website**. For installation, see the
[core demo guide](demo.md). For the research, start with the [project README](../README.md).

## Local preview

From the website repository root:

```bash
python -m http.server 8765 --bind 127.0.0.1
```

Open <http://127.0.0.1:8765/> and <http://127.0.0.1:8765/benchmark/>.
On a remote machine, forward this loopback port over SSH. Do not expose a
research server just to preview the static site. No build step, backend, Node
packages, model weights, or GPU is required.

## Deployment

GitHub Pages publishes the `main` branch root at
<https://sealical.github.io/st-mem/>. Push website changes to `main`, then check
the **Actions** deployment and **Settings → Pages**. `.nojekyll` disables Jekyll.
The live site uses HTML, CSS, and JavaScript with no third-party runtime scripts,
fonts, analytics, or website template. Inline JSON-LD is metadata, not executable code.

Main content files:

- `index.html`: the paper's project page and structured scholarly metadata.
- `benchmark/index.html` and `benchmark/README.md`: SMB description, not a dataset release.
- `README.md`: the research-facing GitHub entry point.
- `citation.bib` and `CITATION.cff`: paper citation, with the paper as preferred citation.
- `assets/`: original styles/scripts and author-supplied paper figures.
- `sitemap.xml`: the two canonical research pages only.

Keep URLs relative for local assets and internal links. The site must work under
the `/st-mem/` subpath. Canonical, Open Graph, Twitter, JSON-LD and sitemap URLs
must instead use the absolute public HTTPS address. If that address changes,
update every one of these references together.

## Content and release boundaries

The paper is **accepted at NeurIPS 2026**, as confirmed by the author on
2026-09-26. Keep that status consistent in the README, project and benchmark
pages, share descriptions, JSON-LD, citation files, and GitHub About.
The citation still identifies the arXiv version and notes the acceptance.
Do not invent proceedings volume, pages, a conference DOI, presentation type,
or an exact acceptance date. Update bibliographic fields when the proceedings
record is available, preserving the arXiv identifier.

The public repository now includes a runnable offline CPU demo. Link to its
verified [installation guide](demo.md), not the private integration repository.
Describe it as a core demo using synthetic observations. Do not present it as a
browser inference service, video perception pipeline, or full paper reproduction.
Run package tests and a clean wheel install when changing demo code.

SMB annotations, evaluation scripts, model weights, and source video are not
distributed by this site. Do not label a description-only page as a downloadable
dataset or invent dataset access, licensing, annotation schemas, or release dates.
Paper-reported measurements must remain distinguished from functional code tests.
Figures remain pixel-identical to the author's originals. Preserve
[their attribution](../assets/README.md) and the separate MIT/CC BY 4.0 boundaries.

## Pre-deployment checks

1. Check both HTML pages at 1440, 768, 390, and 320 px for horizontal page overflow.
2. Check cross-page navigation, anchor links, figures, task tabs, citation copy/download,
   and reading with JavaScript disabled. The abstract must not require a click to load.
3. Parse JSON-LD and XML. Validate `CITATION.cff` against CFF 1.2.0. Keep title,
   DOI, author order, and date consistent with `citation.bib` and the visible page.
4. Ensure no private paths, credentials, unpublished data, fake ORCID identifiers,
   unverified publication details, or unavailable download links enter the public commit.
5. After deployment, check anonymous HTTPS access to both pages, the sitemap,
   BibTeX, and original images. A successful deployment is not proof of search indexing.

## Search Console setup (owner action)

1. Sign in to [Google Search Console](https://search.google.com/search-console/)
   using the site owner's account. Add the **URL-prefix** property
   `https://sealical.github.io/st-mem/`, not a Domain property for `github.io`.
2. Choose the HTML-tag verification option. Have the maintainer add the **exact
   verification meta tag** supplied by Google to the homepage, deploy, and then
   click Verify. Do not send passwords, cookies, or access tokens. A parent
   property already verified by the same owner may cover this path.
3. Submit `https://sealical.github.io/st-mem/sitemap.xml` in Sitemaps. Inspect
   the homepage and benchmark URL, run a live test, and request indexing where offered.
4. Review the indexed status, selected canonical, and search impressions later.
   Requests and sitemaps do not guarantee crawling, indexing, ranking, or AI citations.

The account-wide `https://sealical.github.io/robots.txt` returned 404 on
2026-09-10. No crawler exclusion was found there. A file at `/st-mem/robots.txt`
would not control this origin. Do not change the personal site's root repository
or account-wide crawler rules as part of an ordinary project-page update.

Sources: [site verification](https://support.google.com/webmasters/answer/9008080),
[URL inspection](https://support.google.com/webmasters/answer/9012289),
[robots.txt location and scope](https://developers.google.com/search/docs/crawling-indexing/robots/robots_txt),
[GitHub citation files](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files).
