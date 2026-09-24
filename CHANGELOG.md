# Changelog

- Made pinned JEV the default parent-usefulness text evaluator: one sequential call per complete conversation,
  including extracted printable instructions, with three readiness verdicts and one main issue. Exact-input caching
  avoids repeat charges; failures remain available for human review without another model fallback. The web reader
  compares matching human/JEV verdicts, reveals JEV after rating, saves and exports feedback, and keeps picture
  judgments separate. Shared spending reports include JEV. An explicit runtime setting replaces ordinary answer/thread critics with
  one JEV text review per draft while retaining actual-page visual QA and bounded corrections.

- Added bounded conversation experiments, a shared usage/cost ledger for response, image and review calls,
  and a portable human-review reader with source photos, finished pages, version-bound ratings and exports.
  Follow-ups can request focused reinspection of original photos; illustrated design and review now receive
  recent parent constraints. Added development conversations and an exposure registry that distinguishes
  reused regressions from validation candidates. No-writing validation no longer treats pictured pencils or
  spoken sentences as writing demands. Cost estimates include cache-write charges. Added an optional synthetic
  TypeSafe API probe alongside the conversation evaluator.

- Added an evidence-backed, disabled post-Phase-6 plan for evaluating the `llm-as-a-verifier` pattern on public or
  explicitly de-identified Lab candidates. The plan preserves deterministic hard gates and dual independent release
  review, requires position-balanced criterion-level traces and fail-closed errors, and deliberately adds no runtime
  dependency or production authority.

All notable public changes will be recorded here. The project follows semantic versioning after the first stable release.

## Unreleased

- Preserve worksheet concepts through photo intake, idea buttons, challenge generation and open follow-ups.
  Process up to ten photos in upload order and summarize the complete batch. Semantic button answers now record
  delivery lineage so their next challenge button works. Source context reads recovered attachment observations.
  Added public photo/button conversations and source-image review to the frozen release evaluation. Answer repair
  reviews receive their earlier findings so completed corrections are not silently reversed. Photo inspection retries
  a failed read once and keeps usable actions available for a partially read batch. Removed a keyword rule that
  forced human-convention explanations onto mathematical shape and pattern questions.

- Parent explorations can return several ideas, a brief game or an explanation without a fixed lesson template.
  Conversational follow-ups after a picture preserve the planner's intent and requested depth; generic revision
  keywords no longer force another printable. Added preschool, sibling, two-minute and open-follow-up evaluations,
  including a question after picture delivery, in a frozen 50-capture release run with separate reviewer inputs.
  Slack delivers a phone-viewable PNG before its PDF, with alt text and duplicate-delivery protection. Screen layouts,
  clearer construction steps and retained artwork during layout repair reduce the work left for parents.

- Generalized live picture-page design across artifact topics. Related scenes now share one generated artwork sheet;
  code crops reusable subjects and composes labels, exact counts, card boundaries and blank work spaces. Layout-only
  revisions preserve image bytes and enlarge pictures by a measured amount. A compact page contract replaces the
  legacy lesson-plan draft on this route, and repairs inspect the rendered page without losing earlier feedback.
  Added fresh visual transfer cases, checks for missing construction steps, and evaluator calibration for nonreading
  and pointing activities. Supervision requirements on picture pages follow the child's actual instructions.
  Page space follows actual text, picture and drawing needs; heading-only rows no longer shrink the artwork.
  Fixed missing game instructions on "another idea" revisions, build requests routing as short factual answers,
  "just saving" notes receiving suggestion menus, and observation language or unavailable tools causing hazard vetoes.

- Picture pages now use the configured image model for their main subject artwork, with code-rendered text, counts and work spaces. Images are cached across layout revisions and inspected in the combined page review. Added a visual-quality eval supplement; previous vector-page scores do not validate generated-art quality.

- Added live parent-usefulness evaluation across 24 synthetic everyday situations, with real service conversations,
  delivered PDF/page inspection, model-call and latency capture, separate judge inputs, and masked paired comparison.
  Added a reusable picture-page representation for nonreading clues, cards, counted objects and blank work areas;
  routing preserves semantically requested materials, and revision generation/review sees the previous rendered page.
  Picture requests no longer silently fall back to generic text sheets. Removed ordinary teaching-template vetoes,
  fixed ordinary paint, outdoor play, sock matches and unavailable scissors being mistaken for hazards, and corrected different-game and short-answer revisions. Transparent outlines no longer trigger a second generation; picture revisions use measured printed sizes. Six focused
  utility regressions complement live evaluation; model ratings do not establish parent approval or child learning.

- Rebuilt the first three parent-used challenge paths as hybrid visual pages: one high-quality generated illustration
  supplies warmth while deterministic code owns instructions, concept/action diagrams, and child response areas. Ice
  pages no longer reveal the predicted shape, air pages show the required cup orientation, and paper-airplane pages
  isolate the outer wing-tip change. Family-free image prompts, private prompt/model caching, composed-page usefulness
  review, automatic bad-art fallback, and clearer Slack progress make the path faster to retry and safer to evaluate
  on whether a family can actually use it.
- Reset the beta to a trusted-parent threat model and introduced executable proportional validation tiers for ordinary
  parent assistance, child-usable material, and durable/external action. The mandatory safety path now contains eight
  focused child-material cases instead of 1,829 adversarial-input cases; the entire former corpus remains under
  `evals/history/safety/` but is excluded from active Lab discovery and release manifests. The deep plan now treats
  exhaustive parsing, tamper chains, external high-water/multi-root authority, process-kill matrices, and duplicated
  Phase 5/6 promotion machinery as measured retirement candidates rather than permanent product scope. Every active
  suite is tier-owned; the mixed regression suite uses the strongest tier and cannot be omitted from current review.
- Closed the Round 52 convergence findings with 24 public adversarial regressions. Provider-free routing now covers
  urge, unshakable-desire, waking, consumed-by, perfect-aspect, and fictional-continuation self-harm disclosures.
  Shared candidate gates reject advisory, informational-only, personal-matter, discretion, and explicit-ignore
  warning language across every delivery and persistence surface while preserving warnings that continue in force.
  Wet-appliance unlimited-time language, edge and opposite-corner gas controls, and sharp-tool member selections now
  fail closed; worksheets, labels, and emoji choices remain available as represented material. Phase 5 and Phase 6
  fixture accounting remains exact.
- Closed the Round 50 convergence findings with 27 public adversarial regressions. Provider-free routing now binds
  experiential, perfect-aspect, persistent, and fictional-continuation self-harm disclosures. Shared candidate gates
  reject nonbinding, nonmandatory, optional-compliance, and personal-discretion warning language across every delivery
  and persistence surface while preserving fully mandatory warnings. Wet-appliance unrestricted-time language,
  composed and reverse-attached gas-control references, and sharp-tool partitive or remaining-member references now
  fail closed; flashcards, drawings, pictures, icons, and symbols remain available as represented choices. Phase 5 and
  Phase 6 fixture accounting remains exact.
- Closed the Round 48 convergence findings with 52 public adversarial regressions. Provider-free routing now covers
  natural copular, reflexive, nominal, and fictional-to-real self-harm disclosures. Shared candidate enforcement
  rejects compositional warning optionality across every delivery and persistence boundary while accepting protective
  warning predicates in either conjunction order. Wet-appliance delay, last-but-one and diagonally opposite gas
  controls, and sharp-tool pair/trio or additional-member references now fail closed; compound picture/label/word
  targets and represented-tool anaphora remain available. Phase 5 and Phase 6 fixture accounting remains exact.
- Closed a post-Round-46 validation gap with 26 public sharp-tool selector regressions. Quantified named selections,
  named partitives, and unresolved partitives now bind to hazardous tools across provider-free routing and every
  candidate boundary, while explicitly named labels, represented tools on cards or drawings, and ordinary next-step
  choices remain available. Phase 5 and Phase 6 fixture accounting remains exact.
- Closed the Round 44 convergence findings with 50 additional public adversarial regressions. Provider-free routing
  now covers perfect copulas with ordinary duration adverbs, progressive first-person longing, and reflective or
  tempted adoption of fictional self-harm plans. Shared candidate enforcement rejects warnings framed as voluntary,
  merely advice, take-or-leave, or up to the child across response, visual, extension, artifact, delivery, and
  persistence boundaries while preserving compound warnings whose obligations remain binding. Wet-appliance delay
  phrasing, final and landmark-relative gas controls, and singular unresolved sharp-tool selections now fail closed;
  explicitly urgent shutdowns, represented controls, and named safe selections remain available. Phase 5 and Phase 6
  fixture accounting remains exact.
- Closed the Round 43 convergence findings with 50 additional public adversarial regressions. Code-owned routing now
  covers outcome-noun wishes, perfect-aspect copulas, `long` and `yearn` intent, and completed reflection that adopts a
  fictional self-harm plan. Shared candidate enforcement rejects warning-as-recommendation or formality claims,
  explicit loss of obligation, forget/override permission, and bare decision legitimation across response, visual,
  extension, artifact, delivery, and persistence boundaries, while preserving compound warnings that remain in force.
  Wet-appliance delay, relative gas-control aliases, and unresolved `another`/`one more` sharp-tool references now fail
  closed with urgent shutdown, represented controls, and named safe objects preserved. Phase 5 and Phase 6 fixture
  accounting remains exact.
- Closed the Round 42 convergence findings with 50 additional public adversarial regressions. Provider-free routing
  now covers open grammatical modifiers on active first-person self-harm intent and perfect-progressive
  fictional-to-real follow-through. Shared candidate enforcement rejects warning obligation loss, exhausted-purpose
  claims, and decision deference across response, visual, extension, artifact, delivery, and persistence boundaries,
  while authenticating bounded trusted-adult handoffs and still-binding protective compounds. Wet-appliance delay
  semantics, positional gas-control aliases, and unresolved open-set sharp-tool references now fail closed with dry,
  represented, and named-safe-object contrasts preserved. Phase 5 and Phase 6 fixture accounting remains exact.
- Closed the Round 41 convergence findings with 50 additional public adversarial regressions. Code-owned routing now
  covers nominal and copular first-person wishes, preferences, and outcome intent plus reflective fictional-to-real
  follow-through. Shared candidate enforcement rejects loss of warning force, guidance-only status, disregard freedom,
  and warning set-aside language while authenticating mandatory compound protections. Wet-appliance delay phrasing,
  nearby and end-relative gas controls, and ambiguous `any of them` sharp-tool references now fail closed across source,
  delivery, and persistence boundaries, while named safe plurals and explicit protective reversals remain available.
- Closed the Round 40 convergence findings with 50 additional public adversarial regressions. Provider-free routing
  now covers ordinary preference, ambition, time-qualified goals, and nominal fictional-to-real self-harm intent.
  Shared candidate enforcement recognizes warning expiration, advisory status, passive disregard, and decision
  endorsement while preserving stronger replacement safeguards, mandatory warnings, prevention nonintervention,
  and help-seeking choices. Wet-appliance shutdown discouragement, relative gas-control aliases, and universal
  sharp-tool references are enforced across source, delivery, and persistence boundaries with safe referents intact.
- Closed the Round 39 convergence findings with 50 additional public adversarial regressions. The code-owned safety
  boundary now covers nominal goals and high-confidence fictional follow-through, warning lapse/force/optionality and
  nonintervention polarity, delayed wet-appliance shutdowns, positional and quantified gas-control references, and
  open-class unresolved collective references to introduced sharp tools, while preserving explicit prevention,
  mandatory-warning, safe-rewrite, immediate-shutdown, represented-control, and introduced-safe-object contrasts.
- Closed the Round 38 convergence findings with 50 additional public adversarial regressions. Provider-free routing
  now binds structural first-person intent and fictional-to-real follow-through, deferred or discouraged wet-appliance
  shutdown, ordinal gas controls, and ambiguous display/assortment/collection references. Candidate guards cover
  nonintervention, assent, and warning-nullification forms while authenticating stronger protective prohibitions and
  explicit warning obligations. The Lab artifact boundary now proves every code-owned safety rejection class across
  persistence and delivery instead of assuming candidate probes contain self-harm text.
- Closed the Round 37 convergence findings with 54 additional public adversarial regressions. Code-owned routing now
  recognizes nominal first-person death intent, active death-plan construction, adverb-modified fictional
  follow-through, negative-necessity and broader adult-directed wet-appliance instructions, and marked gas-control
  aliases. Shared candidate guards distinguish additional active, passive, and nominal nonintervention or
  warning-nullification forms from explicitly protective warning/prevention clauses. Sharp-tool resolution now fails
  closed for reversed coordination, whole-set, iterative, and count-based group references across source and artifact
  boundaries while preserving represented-object controls.
- Closed the Round 36 convergence findings with 44 additional public adversarial regressions. Provider-free
  self-harm routing now covers broader first-person death planning, immediate and calendar timing, and
  intent-bearing fictional follow-through. Candidate safety rejects structurally varied nonintervention,
  decision acceptance, and stale-warning reversals across every guarded and persistence surface while preserving
  prevention and still-valid-warning contrasts. Wet-appliance handling covers optional or skipped shutdown and
  broader adult-directed continued-power forms; gas-control aliases cover hash-numbered, landmark-relative, and
  offset controls. Distributive artifact resolution fails closed across coordination and open-ended relational
  phrases while retaining independently predicated ordinary objects.
- Closed the Round 35 convergence findings with 62 additional public adversarial regressions. Source safety now
  binds broader temporal commitments and structurally links fictional self-harm setups to real first-person intent.
  Candidate validation recognizes semantic nonintervention and warning invalidation while authenticating prevention
  efforts and retained warnings before contextual rejection. Wet-appliance handling covers negative necessity,
  adult-directed shutdown prevention, and continued electrification; ignition handling binds gas appliances to later
  possessive controls and additional relative aliases. Distributive artifact resolution now accepts bare plural
  ordinary objects while failing closed when relational supply adjuncts follow a hazardous grammatical head.
- Closed the Round 34 convergence findings with 33 additional public adversarial regressions. Provider-free
  self-harm routing now covers immediate event-relative and onward commitments plus broader fictional setup and
  real-world follow-through language. Candidate safety rejects no-attempt and unwilling nonintervention, decision
  interference waivers, modal permission, and annulled or obsolete warnings across every guarded and persistence
  surface while preserving prevention-effort and still-relevant-warning contrasts. Wet-device safety recognizes
  negative necessity, named-appliance shutdown prevention, and requests for adults to leave devices energized.
  Gas-control handling now includes gas cookers, next controls, relative ordinals, and post-noun proximity aliases.
  Distributive references bind to the grammatical plural head so inserted supplies cannot launder knives while
  ordinary `the`, `some`, and `multiple` paper-card decoration remains valid.
- Closed the Round 33 convergence findings with 47 additional public adversarial regressions. Hazard binding now
  resolves distributive `each one` references to the nearest confident plural antecedent and keeps chemical, sharp,
  and electrical carry actions fail closed without regressing adult-cut/child-decorate artifacts. Provider-free
  self-harm routing covers `upon`, `right after`, and `through` commitments plus determined, act-like, and
  carry-out-the-choice fictional follow-through. Candidate safety rejects consent, favor, nonintervention,
  occurrence permission, and active or passive warning removal on every guarded and persistence surface while
  preserving prevention and retained-warning contrasts. Wet-device safety covers object-bearing shutdown actions,
  `get`/`manage to` inverse shutdown forms, and `taking in` power, while colon-only represented prompts stay benign.
  Gas-control aliases now cover later ordinals, setting modifiers, and immediate/direct proximity.
- Closed the Round 32 independent convergence findings with 51 additional adversarial regressions. Provider-free
  routing now recognizes bounded event-relative death commitments, broader nominal resolve, and determined,
  contracted, `do as`, or emulate follow-through after fictional self-harm context. Candidate validation rejects
  approval grants, explicit inaction, contextual permission, and warning waiver on every guarded and persistence
  surface while preserving emphatic condemnation and continuing opposition. Wet-device safety models positive
  energized-state permission and obligation, shutdown-prevention contractions, and current-draw synonyms while
  excluding represented worksheet instructions. Ignition safety binds modifier-rich appliance controls, quantified
  aliases, and mediated operations while retaining adult, negated, represented, explicit non-contact, and
  adult-cut/child-decorate contrasts.
- Closed the Round 31 independent convergence findings with 50 additional adversarial regressions. Provider-free
  routing now binds structurally varied calendar commitments and real-world imitation after fictional self-harm
  context. Candidate validation treats withdrawn opposition, stated inability or unwillingness to intervene,
  extended blessings, and lifted warnings as unsafe on every guarded surface while preserving condemnation,
  opposition, and prevention advocacy. Wet-device safety normalizes ordinary contractions, continued power receipt,
  live/energized states, and adult or pronoun-bound power-loss prevention without executing represented sentences.
  Ignition safety covers ordinary stove, oven, and burner control heads, modified aliases, and mediated placement,
  with relation-bound non-contact contrasts that cannot launder physical contact operations.
- Closed the Round 30 independent convergence findings with 50 additional adversarial regressions. Provider-free
  routing now covers broader temporal death commitments, reflexive devotion, and chosen, resolved, progressive, or
  footsteps continuations after fictional self-harm context. Candidate validation rejects blessing, non-intervention,
  and active or passive warning invalidation across response, visual, extension, artifact, and persistence boundaries,
  while preserving explicit opposition and prevention advocacy. Wet-device analysis covers inverse shutdown rules,
  power-loss prevention, colloquial or possessive restoration, mains-power supply, and continued power draw, while
  excluding explicitly pictured or transcribed wording. Gas-control analysis normalizes possessive, modified, and
  handle aliases and authenticates bounded non-contact observation without letting contact actions borrow that escape.
- Closed the Round 29 independent convergence findings with 50 additional adversarial regressions. Provider-free
  routing now covers dedication and vow commitments, time-modified death outcomes, plural elections, bare-semicolon
  and natural follow-the-character continuations, while preserving explicit chapter-character and modified
  video-game scope. Candidate validation rejects standalone warning cancellation, expiration, retraction, blessing,
  and broader non-intervention language across every persistence boundary while preserving complete prevention
  advocacy. Wet-device analysis covers possessive, passive, mains-power, and colloquial restoration, adult shutdown
  prevention, and continued power draw while rebinding pronouns to nearer introduced objects. High-risk control
  analysis carries determined aliases and `below`/`between`/`underneath` mediated relations while preserving explicit
  non-contact observation.
- Closed the Round 28 independent convergence findings with 50 additional adversarial regressions. Provider-free
  routing now covers singular and plural reflexive death commitments, elected outcomes, colon/em-dash fictional
  continuations, and follow-the-character intent while preserving death-penalty and game-level contrasts. Candidate
  validation rejects blessing, non-opposition, non-intervention, and invalidated safeguards across every persistence
  boundary while preserving prevention advocacy and retained warnings. Wet-device analysis covers broader shutdown
  prevention and power restoration forms without rejecting prevention of re-energization. High-risk control analysis
  carries object-mediated contact across clauses and handles operation-bound `under`/`beneath` relations while
  preserving represented and reflective references.
- Closed the Round 27 independent convergence findings with 50 additional adversarial regressions. Provider-free
  routing now recognizes reflexive commitments and pledges, selections of death, and real `to too`/same-mind
  continuations after fictional context while allowing episode-positioned fiction. Candidate validation rejects
  contextual endorsement, approval, condoning, non-opposition, non-intervention, and passive or possessive safeguard
  reversals across response, visual, extension, artifact, and persistence boundaries; bounded character-directed
  death choices remain fictional. Wet-device analysis covers power restoration, de-energizing, passive shutdown
  prohibitions, and prevention of shutdown while preserving prevention of re-energizing and adult-owned actions.
  High-risk control analysis binds object-mediated contact while preserving negation, adult learning or ownership,
  and represented controls.
- Closed the Round 26 independent convergence findings with 50 additional adversarial regressions. Provider-free
  routing now recognizes determination, singular and plural choices of death, and real same-plan intent after
  fictional context while allowing chapter- or section-positioned fiction. Candidate validation rejects structurally
  directed death endorsements, contextual sanction or leave, and cancellation or withdrawal of safeguards across
  response, visual, extension, artifact, and persistence boundaries. Wet-device analysis covers `shall`,
  obliged/designated/set states, prohibited shutdown, and give/route/pump power transfers, with safe reconnection
  prevention and nearest-entity binding. Gas-control analysis binds adverbial force and body-part contact while
  preserving adult discussion and represented controls.
- Closed the Round 25 independent convergence findings with 43 additional adversarial regressions. Provider-free
  routing now recognizes perfect-form resolutions, singular and plural selection, choice constructions, explicit
  follow-through, and real `too`/`as well` continuations after fictional context. Candidate validation rejects
  contextual authorization or consent and generalized cancellation of warnings, reminders, instructions, and
  safeguards across response, visual, extension, artifact, and persistence boundaries. Wet-device analysis covers
  `is to` state requirements, prohibited shutdown ownership, delivery or supply of power, and safe prevention and
  off-state polarity. High-risk gas-control binding includes `against` and `about with` relations while preserving
  adult discussion. Numbered and Roman-numeral fictional scene positions remain benign without laundering a later
  real disclosure.
- Closed the Round 24 independent convergence findings with 36 additional adversarial regressions. Provider-free
  routing now recognizes opted/intent-on and choice-noun commitments, plural suicide choices and commitments, and
  real `follow suit` or `so do I` continuations after fictional context. Candidate validation treats caution erasure,
  contextual permission, and anaphoric acceptance as unsafe across response, visual, extension, and artifact fields,
  while preserving `contact a trusted adult` handoffs. Wet-device analysis handles has-got-to/meant-to/passive
  continuation states, negative shutdown ownership, electricity removal, moisture-filled devices, reverse-order
  condensation, and safe off-state polarity. High-risk controls use an open relational-action grammar while preserving
  worksheet and illustration scope, toy appliances, mime, and adult observation. Explicit fictional scopes may carry
  bounded scene-position tails without laundering a later real disclosure.
- Closed the Round 23 independent convergence findings with 46 additional adversarial regressions. Provider-free
  routing now composes present-tense choice and intent, suicide-nominal commitments, direct `that` and `whereby` plan
  complements, and real anaphoric follow-through after fictional context. Candidate validation rejects nonjudgment,
  acceptance, entitlement, warning-erasure, do-not-let-that-stop-you, and concessive warning reversals across
  structured fields while permitting polite and urgent trusted-adult handoffs in either clause order. Wet-device
  state propagation covers kept/required/ought-to states, prevented shutdown, internal moisture, and negated
  unplugging. High-risk control analysis recognizes additional relational manipulation verbs while preserving
  picture/worksheet prefixes, pictured or diagram controls, toy-stove performance, recorded observation, and
  adult-owned operation. Fiction scoping now covers readers, performed monologues, acted character decisions, and
  bounded in-story endings without allowing a later real commitment to inherit the exemption.
- Closed the Round 22 convergence findings with 40 additional adversarial regressions. Provider-free routing now
  composes open commitment nouns and predicates with `where`/`in which` clauses and bounded circumstances. Candidate
  validation recognizes broader support, non-opposition, autonomy, and warning-disregard constructions while
  protective handoffs accept explicit people talking about a safety topic in either clause order. Wet-device state
  propagation covers supposed/required power states, permit/ensure constructions, electricity flow, deactivation,
  and shut-off polarity. High-risk control analysis distinguishes tinker/toy/poke operations from adult observation,
  quoted words, pretend performance, paper props, stories, and controls depicted in images. Fiction and educational
  reference scoping now covers performed characters, drama clauses without punctuation, quoted stage lines, and
  non-intentional reading about a topic.
- Closed the Round 21 independent convergence findings with semantic, cross-surface safety regressions. The
  provider-free floor now recognizes relational self-harm intent expressed through promises, agreements,
  commitments, settled-on gerunds, and where-complement plans; complete character, drama-class, and scene scopes
  remain benign. Candidate validation rejects contextual approval, blessing, non-intervention, preference, and
  explicit warning reversals across response, visual, extension, and artifact fields while preserving action-first
  trusted-adult handoffs. Wet-device handling now covers modal connected/on/plugged states, allow-running and
  maintain-power directions, and negated power removal. High-risk control binding recognizes relational operations
  while preserving adult-owned observation, quoted language, and represented controls. All 27 cases are permanent
  promotion-corpus regressions.
- Generalized the provider-free safety floor from the Round 20 independent convergence review. Owned self-harm
  intent now binds open method nouns, punctuation and gerund complements, modal group agreements, and readiness
  language while preserving complete school-play and story-character scopes. Candidate validation carries self-harm
  context across structured fields and rejects semantic exceptions, recommendations, proceed-anyway directions, and
  autonomy undercuts while allowing emphatic prevention and trusted-adult handoffs. High-risk controls accept bounded
  open noun phrases and additional pressure/loading relations with clock-control rebinding. Wet-device handling now
  covers unhyphenated waterlogged and moist states, continued connection or operation, wall-power language, and
  negated or avoided shutdown actions. All 39 review cases are permanent promotion-corpus regressions.
- Hardened the Phase 2 safety floor with normalized indirect self-harm routing, relation-bound answer-only refusal for
  conductive objects used inside powered appliances, and cross-field response/visual/artifact hazard detection. Typed
  response requirements—including a relation-specific wet/appliance-water prohibition—are bound into each approved
  plan; safety-critical turns are preclassified and construct a
  deterministic code-owned response before any provider call, for bound or unbound events, so model delay or failure
  cannot suppress the minimum answer. This includes conductive-object appliance/outlet actions, energizing an appliance
  around a conductive object, carrying damp/soaked/splashed or passive/floating/in-water appliance state across sentence
  boundaries, including hyphenated/water-logged, intensified or modified submerged, knocked-over/tumbled/dropped,
  passive knocked/fallen, perfect-passive,
  leaking/dripping, and reverse water-inside descriptions, using
  powered devices near or over water, while binding named actions and nearest-antecedent pronouns to the specific wet
  appliance, or
  putting common powered-device and appliance aliases into open-vocabulary water
  containers. Appliance-water remains an independently approved subtype even when a broader electrical category is
  already present,
  fingers or objects entering outlets, ignition and fail-closed gas-valve/knob/dial operations,
  hazardous cleaner handling, and sharp/powered tools, including previously unseen direct control verbs for high-risk
  devices, including later pronoun-, possessive-, and repeated-head-noun operations after an observational mention,
  with nearest-entity rebinding for intervening ordinary objects, while preserving explicitly observational or
  educational-only instructions, including explicit toy/model/radio/thermostat referent rebinding, worksheet
  annotation, cardinal/selective/partitive quantified controls, adverbial force, fastener loading, and
  force/pressure/change nominalized operations.
  Wet-device energizing includes recharge, reactivate, boot-up, reset, resume/continue/keep/carry-on operation, wake, switch-back-on,
  fire-up, charge-up, and power-up wording, plus negated shutdown directions and leave-running/keep-powered state
  language, natural degree phrases, and sprayed-with-water, soaking, and water-damaged states. Self-harm
  intent/aspect variants—including persistent,
  past, planned (including possessive, written-down, completed/finalized, multiply adjective-modified, dated,
  all-ready/set-state, modifier/reordered readiness and timing, and structurally bound singular/plural first-person
  plan, method, step, instruction, strategy, outline, how-method, pact, agreement, and readiness ownership independent
  of the construction verb),
  preparing/chosen, recalled,
  method-thought, intentional-injury,
  and prior-attempt disclosures—action-to-object binding,
  first-person/adult ownership, polarity-aware safety negation and inability-to-avoid/resist/help/stop reversal,
  child/relative/group ownership, fail-closed unresolved pronouns, subject continuity across a nearby adult object,
  child-recipient gender resolution, and nested child reporting verbs,
  explicit adult narrative ownership, bidirectional contextual appliance aliases, explicit alternate objects, secured
  hazards, open-vocabulary safe destinations (including external pronoun relations), and benign compound/idiom
  contrasts prevent both missed and false emergency routes. Candidate-authored self-harm directions and common
  recommendation, desire, permission, deserved/comparative, target-first evaluation, polarity-reversal, next-step, and
  suggestion frames now
  fail closed across response prose, extensions,
  visuals, and artifacts, including generalized suicide/self-harm/overdose nominal endorsements, reverse-predicate
  endorsements, modal/involvement predicates, labeled recommendations, cross-clause imitation, benefit claims, and
  negation-shaped reminders. Self-harm context now spans all child-visible structured fields and rejects adversarial
  protective-then-hostile continuations. Protective prevention/help education is authenticated as a complete clause
  before it is allowed, including preventability predicates, vital/important prevention, prevention education,
  qualified anti-endorsement, and trusted-adult topic
  handoff clauses; danger
  statements, conditional accidental-injury/death warnings, and immediate modal trusted-adult handoffs remain
  deliverable. Clearly scoped whole-statement character, roleplay, writing, and prefix/suffix game/story method
  language remains an
  ordinary contrast rather than an emergency inference. The authenticated
  candidate-boundary Lab case exercises real persisted response rejection for prose, visuals, and extensions, then the
  real artifact generation/persistence service to prove the hostile draft fails for the self-harm reason alone, benign
  controls pass on their first attempt, only a safe unreviewed fallback can persist after rejection, and no delivery
  record, evidence, or opportunity is created. Explicit danger warnings and immediate trusted-grown-up handoffs remain
  permitted on answer-only routes. Mixed safe-then-unsafe
  appliance instructions are evaluated action by action. Route telemetry explicitly
  records that no model critic executed. Pedagogy
  contract/validator 2.1 remains backward-readable, while a
  safety-relevant stored 2.0 plan is recompiled into a separately persisted 2.1 plan before generation. The
  promotion-required safety suite now has code-minimum, corpus-authenticated candidate-boundary,
  social-uncertainty, self-harm, and physical-hazard categories, with every typed physical no-go family and benign
  compound/idiom contrasts represented; removing or
  relabeling a required category fails closed.
- Clarified that the Phase 0–6 software candidate is implemented while consented family co-design and the real-data,
  independent-live-judge promotion gates remain incomplete. Synthetic tests cannot be represented as human validation.
- Added the schema-v26 Phase 6 research track as a closed candidate: a fixed, evidence-linked Bayesian uncertainty
  companion and inspectable context-driven proactivity candidates that cannot change learner projections or write to
  delivery queues. Real data requires an exact corpus manifest and independent privacy review; uncertainty applies
  corrections and counts distinct episodes rather than attempts. Promotion binds the current release, requires group-disjoint held-out metrics,
  task/domain/outcome diversity, the full independently judged Lab, a distinct reviewer and operator, operator
  activation, and explicit per-child parent opt-in; synthetic fixtures cannot unlock it. Recovery and every live
  authorization now recompute the calibration evaluation, revalidate the full Lab report, and authenticate promotion
  plus append-only parent-control lineage. Approved calibration is frozen before activation, and all child-scoped
  research tables participate in whole-model backup conflict detection. Activation now creates an immutable
  release/evaluation/review/operator snapshot; recovery reconstructs epoch lineage and re-derives snapshot, trace,
  and candidate IDs, rejects disabled-policy rewinds, and rebinds archived Lab manifests to their review releases.
  Child export includes activation authority before the first trace. Built wheels ship a complete self-verifying
  release manifest and are smoke-tested without a source checkout.
- Hardened Phase 6 authorization with a database-bound external high-water ledger and policy/control event deletion
  guards, so removing revocation events and rewinding database rows cannot resurrect a grant. Installed runtimes now
  reject source-manifest authority from a checkout that does not own the executing package.
- Made that authority ledger crash-recoverable with two-phase grant/revocation resolution, replicated it to registered
  backup roots so stale snapshots inherit later revocations, and added erasure floors so a family can explicitly
  re-enroll after model deletion without restoring prior consent history. Root-first revocation fanout now covers the
  authenticated complete peer-root set and existing managed restores under the shared backup/deletion lifecycle lock,
  so partial multi-root crashes plus active-database loss cannot resurrect authority. Missing, damaged, pending, or
  cross-family recovery metadata fails closed. Privileged research connections reconcile before first use, unopened
  restores have pending revocations completed without blocking later consent or erasure, and obsolete active grants
  are refused after re-enrollment. Retained services require the local ledger on every privileged call, while closed
  restore policies can safely remain closed across skipped grant epochs. Stale restore grants are rejected against all
  recovery-peer epochs before changing authority bytes. Collection pause after opt-out is idempotent.

- Added Phase 5 evaluated personalization: immutable shadow baseline/candidate decisions, typed outcome signals kept
  outside learner evidence, categorical privacy-reviewed replay, deterministic benefit/calibration/fairness gates,
  full-Lab independent-judge promotion reviews, separate operator approval/activation/disable, and reversible exact
  per-child family opt-in.
- Added schema v25 personalization lineage, active-plan authorization rechecks, conservative baseline recovery,
  exact child/event/proposal/baseline/concept binding, canonical policy-behavior identity, complete relational-shape
  authentication and exact trigger repair, finite canonical replay labels enforced by SQLite, exact concept-version
  feedback/projection attribution, monotonic authorization epochs with revoke-during response, artifact, and queued
  response-visual generation
  fallback or fail-closed cleanup, direct/sensitive shadow-only floors, complete
  parent inspection/export, a local dashboard and CLI control surface, and a required nine-case personalization Lab
  suite with observably enacted calibration, independently identified/judged baseline-candidate answer generation, and a
  full release-content manifest.
  Suggested strategy changes remain shadow-only; no policy can self-promote or rewrite code.
- Added the Phase 4 adaptation control plane: scoped and reversible parent memory, human-promoted versioned private
  family recipes with immutable behavioral baseline/candidate evaluations and independent judgment, deterministic
  reference-only long-thread compaction with fresh memory suppression, and durable source-bound background runs with
  fenced attempts, ambiguity-safe delivery, one-shot turn wake, overlap suppression, lineage-bound catch-up-once
  scheduling, and parent-confirmed capability-output delivery.
- Hardened Phase 4 with monotonic parent-memory deletion across backups and crash recovery, legacy preference redaction,
  transient natural memory inspection, inspection-to-child-context redaction, exact Slack outbox lineage with terminal
  and erasure redaction plus restored-queue expiry/disable checks, scoped legacy cleanup, tombstone-safe undo,
  enqueue/readiness/claim authorization fencing across binding revocation for text, visual, artifact, and confirmed
  background delivery, including channel-bound targets, leased compare-and-set worker stages, stale-attempt fencing,
  bounded pre-ticket retry/backoff, terminal post-ticket ambiguity, terminal cancellation across re-pair,
  deletion-safe cross-event memory lineage, exact
  recipe/memory/preference bindings on every session artifact, governed automatic printables with
  no-lineage and deterministic fail-closed behavior, dashboard-visible compactions, durable overlap receipts, and
  race-safe waiter registration after terminal background completion.
- Added schema-v24 authority/audit records, parent-facing memory and recipe CLI commands, live parent-agent memory and
  compaction context, and a promotion-required public adaptation Lab suite. Memory, recipe state, compaction, and
  background completion remain physically and behaviorally separate from learner evidence.
- Added the Phase 3 open family learner model: a child-scoped local inspector with plain-language inference rationales,
  evidence eligibility/provenance, uncertainty, append-only correction history, collection pause/resume, parent-set
  retention deadlines, JSON export, replay-safe source/event deletion, and a confirmed full-model deletion path.
- Added an optional child-friendly co-review page whose responses are explicitly review-only and cannot become evidence
  or change recommendations, plus schema-v21 control/audit records, schema-v22 global replay tombstones, complete local
  event/file erasure, and a promotion-required synthetic family-model suite.
- Added schema-v23 produced-event lineage across the parent-agent harness, binding-independent transport replay
  tombstones, interaction erasure, atomic child-bound event lineage, database-enforced collection pause,
  crash-recoverable race-safe private-file erasure, shared-file ownership, typed event/source/model backup-deletion
  manifests with process-safe snapshot/restore serialization, prepublication restore registration, active-disaster
  restore root inheritance, canonical migration-backup discovery, reciprocal cross-family authority, pre-commit
  integrity and stable-identity validation of every managed database-copy class, required authority registries,
  snapshot-manifest binding, WAL-aware mutable-restore scans, active-database purge protection, fail-closed metadata authentication,
  deletion-quarantine-safe snapshots, and registered-restore cleanup,
  authenticated legacy-root adoption, external internal/transport tombstone import on restore, and conservative
  erasure of legacy or control/audit-only child images,
  required backup-root deletion authority, future replay propagation into retained restores, source/expiry replay
  protection with database-verified globally unique source identity, recursive transfer-plan event-deletion closure,
  pause-stable post-reset collection epochs, and crash-recoverable database-local-first backup registration with a
  root-side pending barrier that blocks restore until deletion authority is synchronized,
  complete legacy learner-state deletion with explicit support preferences preserved, and uncapped exports with
  underlying records, source/definition provenance, retention history, and audit history.
- Kept all learner projections shadow-only; the new controls cannot be invoked by a model, and raw inspector evidence is
  excluded from model inputs.

- Added the Phase 2 pedagogy kernel: typed semantic proposals, code-approved immutable pre-generation plans, a reviewed
  K–2 strategy/domain registry, selected instruction disclosure, conservative grade-baseline challenge, accessibility
  and healthy-closure policies, and code-owned mandatory critic floors.
- Added schema-v19 plan lineage across responses, compiled artifact actions, experiences, artifacts, response visuals,
  and attributable feedback, plus a schema-v20 ownership repair for complete plan/event/child deletion; arbitrary legacy
  artifact payloads can no longer execute and release creation never counts as learner evidence.
- Added promotion-required pedagogy, accessibility, and evidence-validity Lab suites plus adversarial plan-authority,
  crash-replay, cross-binding, direct/sensitive, artifact/visual, and zero-evidence tests.
- Hardened Phase 2 against coordinated direct-answer opt-outs, concurrent artifact-plan races, plan-bound event mutation,
  hazardous child directions in any response field, keyword-only or self-confirming evidence, unrelated color cues,
  ambiguous writing-only accessibility claims, cross-event artifact supersession, and inaccessible deterministic
  fallbacks. Added fail-closed detection for hazards introduced after planning, explicit ignition/electrical/razor
  categories plus common hot/powered/open-water equipment, all-visible-field scanning, cross-step pronoun-aware adult
  ownership, a dedicated self-harm phrase-family boundary, answer-only modal/invitational detection, and unsupported
  understanding-claim rejection. Hazard referents are now resolved independently of field order; directive modal forms
  and suggestion frames such as should/must/will, consider, and why-not are blocked from answer-only responses; concrete
  hazard-category actions remain bound without depending on a closed pronoun list; a fixed same-start printable safely
  compares neck length and reach without climbing or stretching.
- Added a schema-v18 evidence spine with versioned concepts, tasks, per-concept rubrics, immutable learning plans,
  observed opportunities, retention expiry, normalized lineage, append-only corrections, replay-safe deletion
  tombstones, and deterministic knowledge/interest projections.
- Kept legacy graph state and model annotations quarantined; generated output cannot evidence itself, and Phase 1
  learner snapshots are physically removed through one bounded runtime/Director input path while personalization remains
  in shadow mode.

- Replaced command-shaped question handling with a bounded conversational manager that chooses parent chat, context
  capture, quick reviewed answers, deep exploration, revision, artifacts, or confirmed scheduling.
- Added a concise quick-answer contract with the same factual, pedagogy, and context critics as full learning threads.
- Added context-only provenance so teacher emails, photos, and observations can source requested artifacts without
  becoming evidence of child interest, readiness, or mastery.
- Added an owner-only Slack form for creating empty private child profiles without sending profile fields to a model.
- Added durable parent-chat sessions, reviewed capability/skill loading, typed policy-checked tools, and semantic Block
  Kit interactions with bound one-use choices.
- Added distinct worksheet, hands-on activity, and mystery-challenge contracts with one-page rendering, preview, QA,
  and ordered Slack PDF delivery.
- Added explicitly confirmed weekly parent check-ins, durable schedule runs, and restartable Linux user services.
- Added a no-clone installer and private Codex/Claude Code setup handoff under `~/.curiosity-engine`.
- Improved thread-aware revisions, playful output guidance, visual delivery resilience, and malformed graph-update recovery.
- Added paired Slack photo context with bounded vision inspection, durable private thread memory, and proportionate
  parent replies that do not force a full lesson or visual for every turn.
- Added Slack status reactions that change from `:eyes:` while a reply is processing to a persistent
  `:white_check_mark:` when it is done.
- Added a bounded conversational parent-agent loop with current-thread output recall, validated revision targets,
  and explicit reversible thread preferences that never become child evidence.
- Added deterministic child-name routing for strong single-name signals, retained ambiguity prompts, and kept rejected
  drafts from becoming the thread's revision target.
- Added a reviewed before-and-after ice-expansion diagram path for visual-only revisions and removed false-success
  fallback copy after rejected revisions.

## 0.1.0 — initial Slack MVP

- Added local-first household setup, exact Slack pairing, fixed transport proof, unassigned capture, and durable delivery handling.
- Added bring-your-own OpenAI, Anthropic, and OpenRouter brain routes with private configuration and family-data-free probes.
- Added an elementary family lens, bounded licensed-resource retrieval, one-page paper artifacts, and parent-controlled printing.
- Added episode-aware context evidence, repair/retry grouping, inspectable graph health, and append-only parent corrections.
- Kept embeddings, a graph database, probabilistic mastery, and context-driven unsolicited suggestions outside this release.
- Added deterministic tests, behavioral Lab suites, public-project vetting, privacy guidance, and open-source contribution guardrails.
- Added pinned GitHub Actions, CodeQL scanning, Dependabot updates, and an explicit first-publication safety checklist.
- Hardened same-origin local-console redirects, credential-template updates, and terminal secret redaction after the first hosted security scan.
- Added checksum-verified, owner-only family snapshots with credential exclusion and non-destructive recovery.
