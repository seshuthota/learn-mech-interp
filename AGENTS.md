# AGENTS.md

## Commands
- `npm start` — dev server with hot rebuilds (runs validation)
- `npm run build` — production build to `_site/` (validation + Pagefind indexing)
- `SKIP_VALIDATION=1 npm start` — bypass build validation for draft work
- No test, lint, or typecheck commands exist; validation is build-time only.

## Content editing (read these before touching articles)
- `ARTICLE_GUIDELINES.md` — tone, structure, formatting, figure sourcing
- `CLAUDE.md` — article-vs-paper distinction, when to read ARTICLE_GUIDELINES.md
- `CONTRIBUTING.md` — step-by-step article/block creation workflow, frontmatter spec, shortcode reference

## Architecture
- **Articles**: `src/topics/<block-slug>/<article-slug>/index.md` with Nunjucks frontmatter + Markdown body.
- **Flat URLs**: article at `src/topics/probing/probing-classifiers/index.md` serves at `/topics/probing-classifiers/` (block slug omitted from URL).
- **Images**: place in `src/topics/<block>/<article>/images/`, reference at `/topics/<article-slug>/images/<file>`. Build remaps nested paths to flat output.
- **Computed data** (`glossary.js`, `learningPath.js`, `gitMeta.js`) in `src/_data/` — do not edit directly.
- **Validation** runs on every build (incl. dev). Checks: required frontmatter, contiguous block/article ordering, valid citation keys, valid prerequisite links, no duplicate glossary terms or references.
- **Pagefind** search indexing runs automatically as an Eleventy `after` hook.
- **Shortcodes** defined in `eleventy.config.js`: `{% cite "key" %}`, `{% sidenote "..." %}`, `{% marginnote "..." %}`.
- **Nunjucks** is the template engine; Markdown uses markdown-it with KaTeX and figure plugins.
