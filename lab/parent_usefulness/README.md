# Parent usefulness: live service evaluation

The primary measure is whether a parent with small kids could use the delivered
help as-is, in the situation they described. Evaluate ordinary play, spoken answers,
siblings, scarce time/materials, nonreading picture pages, and revisions. A valid
schema or an attached PDF is not evidence of utility.

This complements the offline Lab. It uses the real configured model and
`CuriosityService.chat`, registers a disposable synthetic household per case, feeds
follow-ups through actual history, and captures the messages and files that the
service would deliver. It never starts a Slack transport. Slack buttons, uploads,
acknowledgments and network latency require a separate transport check.

## Run and review

Use the application's Python environment with its existing dependencies. These
commands make billable model calls through the selected configuration. Inputs are
synthetic; no real family conversation or database is used. The auth directory
contains `brain.json` and `model.env` for generation and an owner-only `typesafe.env`
with `TYPESAFE_API_KEY` for JEV evaluation. It is read, never copied to the eval pack.
Store generated results beneath ignored `private/`. PDF text extraction and the
human reader use Poppler's `pdftotext` and `pdftoppm`.

```sh
python lab/parent_usefulness/collect.py \
  --source /absolute/source --auth-root /absolute/private/setup \
  --run-dir /absolute/private/evals/candidate --budget-usd 5 \
  --usage-db /absolute/private/evals/usage.sqlite \
  --inputs lab/parent_usefulness/candidate-inputs.json

python lab/parent_usefulness/review.py \
  --source /absolute/source --auth-root /absolute/private/setup \
  --run-dir /absolute/private/evals/candidate --budget-usd 5 \
  --usage-db /absolute/private/evals/usage.sqlite

python lab/parent_usefulness/human_review.py \
  --runs /absolute/private/evals/candidate \
  --output /absolute/private/evals/web

python lab/parent_usefulness/summarize_human.py \
  --page /absolute/private/evals/web --reviews EXPORTED_RATINGS.json \
  --output /absolute/private/evals/human-summary.json
```

All runners accept `--ids PU01 PU02 ...`. Collection defaults to two concurrent
cases and requires the live OpenAI backend for SDK usage telemetry. Keep source
fixed throughout a run. Each bundle records its source fingerprint, configured
model routes, actual turns, file hashes, latency and model calls. Synthetic draft diagnostics explain repairs but are excluded from utility judgment. Report unknown
timing/cost as unknown. Existing captures and reviews are preserved; use a new run
directory for a fresh sample. Service failures are product failures, including any
dependent turns that cannot proceed. Capture failures remain not reviewable.

The generator receives only `candidate-inputs.json`. It never receives the rubric,
case expectations, calibration or split. JEV (`jev-1.13.0`) is the sole default text
reviewer: one sequential call for the whole conversation, including printed PDF
instructions, returns **Ready / Needs edits / Wouldn't use** and one main issue.
There is no OpenAI shadow judge, extra threshold, retry or model fallback. Errors
remain available for human review. Unchanged inputs reuse their cached JEV result.

Open `web/index.html` to rate text with the same three choices. JEV appears beside
your verdict after you rate; optional reason buttons and keyboard shortcuts keep
review quick. Check source photos and finished pages with the separate **Pictures**
buttons: JEV only reads text. Export ratings to compare agreement, disagreements,
and acceptance after picture review. Model ratings are a proxy; never convert them
into claims about actual child engagement or learning.

## Development discipline

Start with a baseline on the current release. Inspect real pages before changing
the harness. Fix recurring work left to the parent—having to invent the activity,
draw the page, repeat a request, or wait through unnecessary reviews—then rerun the
16 development cases. Report individual losses alongside totals and latency.

The initial eight reserved cases have already been sampled in the September 2026
baseline. Any disclosed example used to guide a change becomes a regression, not
an unseen holdout; in particular, the missing PU23 counted-object page and PU21 story-card failure were disclosed.
Use fresh subject families for later generalization checks and freeze the candidate
before inspecting them. This small pack does not establish coverage of all prompts.

Keep ordinary factual, accessibility and concrete-risk checks in the normal Lab.
Passing those checks does not compensate for an unusable parent experience.

## Files

- `cases.json`: 24 synthetic scenarios, input/reviewer separation and original split.
- `candidate-inputs.json`: reviewer-free input mapping for generation.
- `rubric.md`, `calibration.json`: use-as-is rubric and example judgments.
- `collect.py`: live service capture and usage/latency measurement.
- `review.py`: fast JEV text classification, PDF text extraction, exact-input cache.
- `human_review.py`, `human_review.html`: saved side-by-side human/JEV ratings and separate picture checks.
- `summarize_human.py`: acceptance and human/JEV comparison from exported ratings.
- `compare.py`: legacy optional OpenAI paired diagnostic; excluded from the default flow.

## Generated visual quality

The original vector-page usability result does not establish illustrated quality. Use the [illustrated supplement](illustrated/README.md) to inspect recognition, appeal, explanatory content, label placement, counted objects and revision continuity on actual generated pages. Report image API calls and waiting time separately; attractive artwork can still depict the wrong story or relationship.

## Frozen complete release run

`python lab/parent_usefulness/run_release.py --source RELEASE --auth-root PRIVATE_SETUP --run-dir NEW_RUN --budget-usd 5` evaluates all six packs (50 conversations) and six predeclared repetitions on one unchanged runtime. Reviewer expectations are frozen separately from the generator inputs. JEV text review and the human reader are included by default; `--capture-only` skips JEV, and `--typesafe-key-file` overrides its credential location. Report complete text ratings, human picture judgments, timing and failures; a zero exit status means the run completed with consistent hashes, not that every output was useful.

The twelve [everyday and open-follow-up cases](everyday/README.md) are development cases for parents of small kids, including phone use, siblings, concise games, simple construction, selective artwork edits, natural references and topic changes. They contain synthetic wording and no private Slack transcript.

Photo/button cases are documented in [photos/README.md](photos/README.md). Their input images are original public fixtures; real family photos and replay bundles stay private. Source photos are shown separately from delivered pictures to the reviewer.

See [complete harness experiments](../../docs/HARNESS_EXPERIMENTS.md) for shared spending limits, source-aware follow-ups, portable human review, and validation exposure tracking.
