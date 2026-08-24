# Curated public projects and supply-chain policy

Curiosity Engine points to canonical upstream projects instead of copying snapshots into this repository. A link is not permission to install, execute, copy content, or send family data. The default for every external project is **reference only** until a separate integration review promotes a particular stable release.

The machine-readable catalog is `configs/public-projects.json`. It records the canonical source, publisher, license, freshness feed, age-fit limits, safety notes, and one of four statuses:

| Status | Meaning |
|---|---|
| `integrated` | A specific released dependency is already used, constrained in `pyproject.toml`, and covered by public tests. |
| `approved_reference` | Agents may consult its public documentation and released design patterns. They may not clone, install, import, or execute it automatically. |
| `evaluation_candidate` | Relevant and credible, but dependency size, parser risk, model downloads, licensing details, or child fit still need a dedicated evaluation. |
| `watch_only` | Useful ecosystem signal, but not acceptable for MVP execution or integration. |

## Researched shortlist

This is a source map, not a grab bag of dependencies.

| Need | Projects | Current conclusion |
|---|---|---|
| Deterministic paper artifacts | ReportLab; Typst; WeasyPrint | ReportLab remains the integrated renderer. Typst and WeasyPrint are references for template and paged-layout ideas. WeasyPrint needs an SSRF-safe URL-fetch boundary before any integration. |
| Visual OCR and scanned worksheets | OCRmyPDF; PaddleOCR; Tesseract | OCRmyPDF is a strong local pipeline reference. PaddleOCR is active and capable but has a large code/model-download surface. Tesseract is a useful baseline, but its reviewed security-support table appears stale. None is installed by onboarding. |
| Worksheet/question generation | Numbas; SymPy | Numbas offers university-governed schemas, marking, and parameterized-question patterns. SymPy may eventually verify allowlisted arithmetic, but must never evaluate untrusted model/parent strings. Neither supplies elementary pedagogy. |
| Learning interactions | H5P Memory Game; PhET; Blockly | H5P offers a small accessible game pattern. Use only published PhET simulations because PhET says repository main is unstable. Blockly's canonical repository is now maintained by Raspberry Pi Foundation; production use must select a release rather than main/beta. |
| Game generation | Phaser; GDevelop; Scratch Editor | Phaser is a strong engine reference, not a pedagogy. GDevelop is an active but very large extension platform and remains evaluation-only. Scratch is watch-only during its monorepo transition and brings AGPL, trademark, account, and age-fit considerations. |

Open the current catalog without contacting any external service:

```bash
curiosity ecosystem status
curiosity ecosystem list --category visual_ocr
curiosity ecosystem show --id phet
```

Refresh only public metadata explicitly:

```bash
curiosity ecosystem check --live
```

The live command contacts GitHub and PyPI metadata APIs. It sends no family data, downloads no upstream source, and executes nothing. It detects obvious changes such as archival, a disabled repository, unexpected license metadata, a stale last push, or a yanked package. It is an alarm, not an approval decision.

## Vetting gate

An agent must complete all four reviews before proposing an integration.

### 1. Identity and legal review

- Resolve the canonical repository from the publisher's official site; do not trust a similarly named fork or package.
- Confirm an OSI-approved software license in the exact repository and release being considered.
- Review licenses separately for code, curriculum text, templates, images, fonts, audio, models, sample data, plugins, and trademarks.
- Publicly visible source with no license or “all rights reserved” content is not reusable.
- Record relocations, ownership changes, archived repositories, and package-manager publisher identity.

### 2. Maintenance and security review

- Require meaningful recent maintenance or a clearly supported stable branch—not stars alone.
- Inspect release cadence, signed release evidence where available, CI/tests, multiple maintainers or credible institutional governance, issue handling, and a vulnerability-reporting route.
- Check current advisories and transitive dependencies with OSV/registry tools. Use OpenSSF Scorecard and Best Practices badges as signals, never as sole approval.
- Treat a new major release, maintainer/owner change, license change, archive notice, unexplained binary/model download, or security-policy regression as a mandatory re-review.

### 3. Technical containment review

- Start from a stable release; never execute `main`, a release candidate, a gist, or a remote install script.
- Pin the version and immutable commit/artifact hash. Use lockfiles and verify package/source provenance when supported.
- Inspect install/build scripts before running them. Test in an isolated environment with no Slack/model credentials, no `private/` mount, minimal filesystem permissions, no listening network service, and network denied unless the test explicitly requires it.
- Put parsers around file-size, page-count, pixel-count, time, memory, and subprocess limits. Treat PDFs, SVG, fonts, images, archives, models, plugins, and expression languages as untrusted input.
- Do not enable telemetry, analytics, accounts, ads, purchases, multiplayer, remote plugins, or arbitrary URL fetching for a child/family workflow.

### 4. Educational and family-safety review

- Verify the project is a tool or pattern for the intended age—not merely tagged “education.”
- Evaluate first-grade reading load, instructions, motor demands, time pressure, accessibility, feedback tone, failure states, distraction, and parent effort.
- Generated games must have a real learning purpose and a graceful stopping point; engagement alone is not educational value.
- OCR output is uncertain evidence. Math output needs an independent deterministic answer/key check. Every printable still passes artifact trust, exact layout validation, visual QA, and parent approval.
- Use synthetic/public fixtures first. A candidate never receives family or licensed-resource content during evaluation.

## Promotion and review cadence

1. Add the project as `evaluation_candidate` with primary-source evidence and explicit risks.
2. Run the live metadata alarm and a manual license/governance review.
3. Build a narrow adapter in an isolated branch with a pinned release and no private data.
4. Add malicious/malformed-input tests plus elementary pedagogy and accessibility evals.
5. Review license obligations and generated-output rights.
6. Promote to `integrated` only through an ordinary reviewed dependency change. `approved_reference` alone is never sufficient.

The catalog expires every 90 days. A coding agent must recheck it when `curiosity ecosystem status` says `review_due`, and immediately when a known trigger occurs. Re-review updates the date and evidence; it does not silently upgrade an installed dependency.

## Important exclusions found during research

- The former `scratch-vm` repository is archived and points to `scratch-editor`; old tutorials that clone it are stale.
- Mathigon's public textbook repository describes its course content as “all rights reserved” and targets ages 12–18. It is not an elementary open-content source for this project.
- PhET's own development guide says main branches are unstable. Curiosity Engine may point families to appropriate official published simulations, but does not run source tip or casually copy simulation assets.
- Small “worksheet generator” repositories without current maintenance, security reporting, governance, tests, and clear content rights are not included merely because they have an open license.

## Security tooling references

- [OpenSSF Scorecard](https://github.com/ossf/scorecard) checks observable supply-chain practices but cannot replace judgment.
- [OpenSSF Best Practices Badge](https://openssf.org/projects/best-practices-badge/) is maintainer self-certification and is one supporting signal.
- [OSV-Scanner](https://github.com/google/osv-scanner) checks known vulnerabilities and can run offline; its network mode sends package metadata and hashes, not source code.
- [GitHub dependency review](https://docs.github.com/en/code-security/concepts/supply-chain-security/dependency-review) can reject newly introduced vulnerable dependencies in pull requests.

Do not add automated installation to `curiosity ecosystem`. Its purpose is to make current evidence visible while keeping execution behind a deliberate engineering and parent-safety gate.
