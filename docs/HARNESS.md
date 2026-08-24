# What the Curiosity Engine harness owns

An LLM is a function: input → reasoning → output. The harness is everything around that function that determines what it sees, when it runs, which tools/models participate, what persists, what is allowed, and how quality is verified.

## Bare minimum

```python
while True:
    event = wait_for_input_or_schedule()
    state = retrieve_state(event)
    context = build_context(state, policy)
    result = call_model(context)
    validate(result)
    save_state(result)
    execute_allowed_actions(result)
    log_run()
```

## Curiosity Engine version

```text
EVENT
 ↓
PERSIST RAW EVIDENCE
 ↓
CLASSIFY / ROUTE
 ↓
CONTEXT BUILDER (depth 0–4)
 ↓
REASONING POLICY
 ├─ generator(s)
 ├─ adversarial critic(s)
 └─ selector/revision
 ↓
SCHEMA + POLICY VALIDATION
 ↓
STATE TRANSACTION
 ↓
ALLOWLISTED EFFECTS + PROPOSED ACTIONS
 ↓
RUN LOG + FEEDBACK
```

Raw evidence and the durable job commit before model work. Model calls occur outside database transactions. A final response, validated graph effects, and action proposals commit atomically. Failed/rejected candidates apply no proposed effects.

The autonomous Director runs above these workflows to discover opportunities. It never replaces them and never directly executes an external side effect.
