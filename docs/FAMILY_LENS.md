# Family lens

The model is not the pedagogy. The family lens is a bounded, parent-selected set of learning and practical preferences.

Public defaults favor the child’s real question, noticing before explaining, a small conceptual stretch, productive struggle, low parent effort, short activities, ordinary materials, and cautious interpretation of evidence.

Accept them with:

```bash
curiosity family-lens configure
```

Customize only what matters:

```bash
curiosity family-lens configure \
  --pedagogy "invite a prediction" \
  --theme "nature walks" \
  --activity-minutes 10 \
  --parent-effort very_low \
  --reading-load emerging \
  --material paper
```

The profile stays in the private database. Begin with grade/readiness, practical constraints, and a few themes—not a complete family biography.

## Family resources

Purchased resources stay under ignored `private/resources/`. Ownership means availability, not that a child has seen, understood, completed, remembered, or liked the material. A purchase URL records provenance; it is not permission to scrape, redistribute, or expose content.

New families use `metadata_only`. Explicitly permit small relevant passages in provider requests with:

```bash
curiosity resource mode --mode selected_excerpts
```

Retrieval still requires relevance, and Slack paraphrases instead of reproducing passages.

## Real-answer review

After the synthetic provider probe, ask one real Slack question. Review:

- factuality and direct relevance;
- grade/readiness fit;
- whether it invites noticing, predicting, comparing, or trying;
- realistic parent effort.

Record with `curiosity onboard review`. Any `retry` keeps setup open for tuning.
