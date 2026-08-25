# Public-project policy

`configs/public-projects.json` points to canonical upstream projects; it does not copy them. A listing never authorizes installation, execution, content reuse, or family-data disclosure.

| Status | Allowed use |
|---|---|
| `integrated` | Pinned released dependency already covered by public tests |
| `approved_reference` | Read released docs and design patterns only |
| `evaluation_candidate` | Isolated review with synthetic/public data only |
| `watch_only` | Ecosystem signal; never execute |

## Reviewed landscape

| Need | Current references | Position |
|---|---|---|
| Paper output | ReportLab, Typst, WeasyPrint | ReportLab is integrated; the others are references |
| OCR | OCRmyPDF, PaddleOCR, Tesseract | Evaluation only; onboarding installs none |
| Questions | Numbas, SymPy | Reference only; neither supplies elementary pedagogy |
| Interactions | H5P, PhET, Blockly | Use releases/published simulations, never source tip |
| Games | Phaser, GDevelop, Scratch Editor | Engine references only; size, licensing, and age fit need review |

Inspect the catalog without installing anything:

```bash
curiosity ecosystem status
curiosity ecosystem list --category visual_ocr
curiosity ecosystem show --id phet
curiosity ecosystem check --live   # public GitHub/PyPI metadata only
```

The live check sends no family data, downloads no source, and executes nothing. It is an alarm, not approval.

## Integration gate

Before proposing an integration:

1. Resolve the canonical publisher and exact stable release; never run `main`, beta/RC code, gists, or remote install scripts.
2. Verify the software license plus separate content, asset, font, model, sample-data, plugin, and trademark rights.
3. Review maintenance, governance, security reporting, advisories, release provenance, dependencies, install scripts, telemetry, and network behavior.
4. Pin the version and immutable hash. Test without `private/`, credentials, or family data in an isolated, resource-limited environment.
5. Treat PDFs, images, SVG, fonts, archives, models, plugins, parsers, and expressions as untrusted input.
6. Evaluate elementary reading load, accessibility, feedback, failure states, parent effort, and actual learning value.
7. Add malicious/malformed-input tests plus pedagogy and accessibility evals.
8. Promote to `integrated` only through normal dependency review. Stars, scores, recency, and `approved_reference` status are signals—not approval.

Re-review when the catalog expires (90 days) or ownership, license, archive state, major version, binary/model downloads, or security posture changes. Never automate installation through `curiosity ecosystem`.
