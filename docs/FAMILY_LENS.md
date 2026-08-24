# Configure the family lens

The model is not the pedagogy. Curiosity Engine sends the model a bounded, parent-selected lens that describes how this family wants learning threads shaped.

## Starter lens

The public starter profile in `configs/family-lens.default.json` favors:

- following the child's real question;
- showing and noticing before explaining;
- one conceptual rung above current readiness;
- productive struggle without spoon-feeding;
- low parent effort and short activities;
- ordinary paper, writing utensils, and household materials;
- observations as evidence, never instant durable traits.

Accept it with:

```bash
curiosity family-lens configure
```

Or customize only what matters:

```bash
curiosity family-lens configure \
  --pedagogy "show before explaining" \
  --pedagogy "invite a prediction" \
  --theme "nature walks" \
  --activity-minutes 10 \
  --parent-effort very_low \
  --reading-load emerging \
  --material paper \
  --material "writing utensils"
```

The resulting profile is stored in the private local database. The setup agent should not ask for a complete family biography. Child grade/readiness, practical constraints, and a few current interests are enough to begin; feedback should evolve the context over time.

## Family resources

Resource ownership means the material is available. It does not mean a child has seen, completed, understood, remembered, or liked it.

Purchased resources stay under ignored `private/resources/`. A URL or purchase record is provenance, not permission to redistribute content. The parent must explicitly authorize importing a locally available copy. Public code, tests, logs, issues, commits, and pull requests may contain neither titles/URLs that identify a private purchase nor copied excerpts.

New families start in `metadata_only` mode. A parent may allow small relevant passages in bounded provider requests:

```bash
curiosity resource mode --mode selected_excerpts
```

This is separate from choosing an LLM provider. Retrieval still requires relevance, and Slack output paraphrases rather than reproduces a purchased passage.

## Final quality review

After the synthetic provider probe, send one real question through paired Slack. Review the answer on four dimensions:

1. **Factuality:** it answers the actual question accurately and uses current sources when needed.
2. **Grade fit:** language, concepts, and reading load suit this child now.
3. **Curiosity value:** it creates something to notice, predict, compare, or try rather than closing the question.
4. **Parent effort:** the activity is realistic with available time and materials.

Record the result locally with `curiosity onboard review`. Any `retry` keeps setup open for tuning.
