# Improve the complete parent experience

The unit of evaluation is a conversation: source photos, a useful idea, an open question, material creation, and a revision. Score whether a parent can use the whole delivered experience as-is. Separately record factual problems, missing output, time, and cost. Do not turn successful generation or a model rating into evidence of child learning.

## Run one bounded experiment

Start from a frozen source snapshot. The generator receives only parent inputs; reviewer expectations remain separate. The runner records hashes of runtime, harness, and case pack before execution and verifies them afterward.

```sh
python lab/parent_usefulness/run_release.py \
  --source /absolute/frozen-source \
  --auth-root /absolute/private/setup \
  --run-dir /absolute/private/evals/new-run \
  --pack lab/parent_usefulness/conversations \
  --budget-usd 5 --workers 2
```

The auth directory supplies `brain.json` and `model.env` for generation and an owner-only `typesafe.env` containing `TYPESAFE_API_KEY` for evaluation. `--typesafe-key-file` can point to a separate key file. Add `--capture-only` to collect for human review without JEV. Omit `--pack` for all six packs and the declared repeat cases. Use `--ids` for a selected subset. Every live process requires a positive budget. For baseline/candidate comparisons, pass the same absolute `--usage-db` to every process so they share one total budget.

## Fast text classification

JEV (`jev-1.13.0`) is the sole default text evaluator. One sequential API call per complete conversation answers two native Choice questions: **Ready / Needs edits / Wouldn't use**, and a main issue (**none / request / constraints / effort / content / delivery**). Code maps the verdict to **ready / edit / rework**. These are evaluation labels, not automatic release or Slack delivery authority. There is no extra probability threshold, second judge, automatic retry or provider fallback. An incomplete capture or unavailable evaluator stays **Can’t judge / review**.

JEV sees parent requests, delivered replies, delivery facts and the actual words extracted from PDFs with `pdftotext`. It does not see expected labels or private generation diagnostics. It cannot inspect original worksheet pixels or generated artwork; a text-ready verdict leaves pictures pending human review. Exact-input results are cached in `jev-review.json`. Rerun the same inputs at no extra model cost; changed inputs or criteria require a new classification.

Evaluate existing captures without generating anything again:

```sh
python lab/parent_usefulness/review.py \
  --runs PRIVATE_BASELINE_CAPTURES PRIVATE_CANDIDATE_CAPTURES \
  --key-file private/setup/typesafe.env \
  --budget-usd 0.05 --usage-db PRIVATE_JEV_USAGE.sqlite
```

The three new conversation cases are development cases. For independently prepared validation cases, use `--purpose validation --pack PRIVATE_PACK --exposure-registry PRIVATE_REGISTRY`. The registry keys inputs rather than case labels and records exposure before execution. Reused inputs become regressions. A development case cannot become a validation case just by selecting a command flag. Different hashes do not establish unseen concepts: keep validation cases away from the optimizer, freeze before inspection, and disclose any feedback used in later changes.

## Review the delivered evidence

The runner creates `human-review/index.html` with local copies of source photos, complete replies, every rendered PDF page, and PDF links. Open it in a browser. No online account is required. Read each conversation and choose **Ready**, **Needs edits**, **Wouldn’t use**, or **Can’t judge**. Your verdict appears beside JEV's; JEV is hidden until you rate unless you choose **Show now**. Optional reason buttons use the same categories as the classifier. Keyboard **1 / 2 / 3** rates text; **N** opens the next unrated conversation. Multiple runs use masked variant letters.

Use the separate **Pictures** buttons to check the source concept and finished pages. A correct-sounding explanation alone cannot pass this check. A note is optional. The header counts agreements and both kinds of disagreement: text JEV approved that you would change, and text JEV flagged that you would use.

To rebuild a portable reader from existing captures:

```sh
python lab/parent_usefulness/human_review.py \
  --runs PRIVATE_BASELINE_CAPTURES PRIVATE_CANDIDATE_CAPTURES \
  --output PRIVATE_NEW_REVIEW_FOLDER
```

Votes save in the browser and bind to the exact output fingerprint. Use **Export ratings** for a durable version-3 JSON copy; **Import ratings** restores matching versions. The export records the JEV policy/input version and whether its verdict was seen before rating. Old whole-experience ratings retain their original scope and cannot be imported as text-only votes. Browser testing uses separate synthetic fingerprints and does not prefill real parent votes.

```sh
python lab/parent_usefulness/summarize_human.py \
  --page PRIVATE_REVIEW_FOLDER --reviews EXPORTED_REVIEWS.json \
  --output PRIVATE_HUMAN_SUMMARY.json
```

The summary reports text ratings, separate picture ratings, human/JEV agreement and a three-way confusion matrix. Whole-experience acceptance requires ready text and passing human picture checks when pictures are present. Missing cost or unreviewed output prevents a complete cost-per-success claim. Evaluation spending is reported separately. Incomplete captures cannot be promoted to quality passes. Version-2 exports remain readable by the summary command as older whole-experience reviews.

## Cost accounting

`model_usage.py` meters OpenAI generation and JEV evaluation at their provider boundaries. Set `CURIOSITY_USAGE_DB` to an owner-only private SQLite file to enable accounting outside evals. Without that setting normal runtime behavior is unchanged. `CURIOSITY_USAGE_PHASE` distinguishes runtime, generation, evaluation, and comparison. No prompt, response, image, credential, or family text is written into the cost ledger.

Budgeted experiments reserve conservatively before each call, across workers. Hidden SDK retries are disabled; explicit application retries must reserve again. A failed or usage-less request retains its reservation because its charge is unknown. Budget exhaustion is labeled `budget_stopped`, not a product-quality failure. Changing or removing the limit on an existing budgeted ledger is rejected.

The rate card is dated and linked to official pricing. Costs include reported cache writes, cached reads, reasoning output, and the long-context premium. They are estimates, not reconciled billing. Unknown models and unsupported image configurations cannot make budgeted calls. Image reservations cover the supported 1024/1536 sizes; the provider account's own spending limit is the ultimate billing control. Review the rate card before future runs, and include all evaluation processes in the shared ledger.

## Source-aware follow-ups

The planner receives bounded observations from the current photo lineage. It can set `inspect_source_photos=true` on an existing learning or artifact action when a new request depends on missing or uncertain details. Code resolves original photos inside that session, performs one focused vision call for the exact parent wording, and passes the result into the action. The initial observation remains intact. Failure to inspect is represented as uncertainty, not invented visual facts.

Recent parent requests and active preferences reach picture design and final-page review. The newest correction takes precedence; a topic change can supersede older topic-specific constraints. Semantic learning decisions bypass several legacy phrase-based revision rules, while authoritative context, operational, and artifact-negation boundaries remain enforced.

## Optional TypeSafe diagnostic

```sh
python lab/parent_usefulness/typesafe_probe.py --dry-run --output PRIVATE_PREPARED_PROBE
python lab/parent_usefulness/typesafe_probe.py \
  --key-file private/setup/typesafe.env --output PRIVATE_NEW_PROBE --max-usd 0.01
```

This separate twelve-example synthetic API probe uses pinned `jev-1.13.0`, no retries, and text-only public examples. Its labels are authored expectations, not parent ratings. Its binary half-probability threshold is illustrative; the default conversation evaluator uses the three native readiness choices above. Neither certifies images. Running an eval does not change Slackbot settings; [explicit runtime configuration](BRAIN_SETUP.md#optional-fast-text-review-with-jev) selects JEV for ordinary text critics.
