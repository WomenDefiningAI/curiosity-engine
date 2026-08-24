# Publishing the public repository

This is a maintainer procedure for the first public push and later releases. Publishing the reusable code is separate from approving any family's model setup. Never copy `private/`, `.env`, generated output, local reviews, or purchased resources into a public commit.

## 1. Prove the candidate locally

From the repository root:

```bash
git status --short
git ls-files private .env data output .Codex
ruff check .
pytest
curiosity-lab --repo . --json-out private/eval-report.json
python -m compileall -q src tests
python -m pip check
python -m pip wheel . --no-deps --wheel-dir private/release-wheel
gitleaks git --redact --no-banner
```

The two `git` commands must print no working-tree changes and no tracked private paths. All executable checks and the secret scan must pass. The offline Lab may correctly say `promotion_eligible=false` when its independent live judge was not run; that blocks model promotion, not an honestly labeled alpha source release.

Read the full commit diff and confirm that documentation, tests, fixtures, screenshots, and examples are synthetic. A local family quality review must not be converted into a public fixture unless it can be completely de-identified and independently licensed.

## 2. Confirm the human decisions

A GitHub organization owner must explicitly confirm:

- the exact owner/name and **public** visibility;
- that the commit is intended for worldwide publication under the MIT License;
- the repository description and topics;
- who receives conduct reports if a `CODE_OF_CONDUCT.md` is added;
- who may administer rulesets and security settings.

Creating a repository, adding a remote, pushing, changing organization settings, publishing a release, or opening a pull request is an external side effect. A coding agent must ask before doing any of them.

## 3. Create and push once authorized

First confirm that the destination does not already exist and that GitHub CLI is using the intended organization administrator:

```bash
gh auth status
gh repo view WomenDefiningAI/curiosity-engine
```

The second command should report that the repository is absent for the first publication. After explicit approval, the maintainer may run:

```bash
gh repo create WomenDefiningAI/curiosity-engine \
  --public \
  --source=. \
  --remote=origin \
  --push \
  --description='Local-first, Slack-connected family curiosity harness'
```

Do not initialize the remote with another README, license, or `.gitignore`; those files already exist locally.

## 4. Apply GitHub safeguards immediately

In the repository settings:

1. Enable private vulnerability reporting. `SECURITY.md` links to that private advisory route.
2. Enable secret scanning, push protection, and validity checks where GitHub offers them.
3. Let the first `behavioral-ci` and `codeql` runs complete successfully.
4. Add a `main` ruleset that blocks force pushes and deletions, requires pull requests for later changes, and requires the `test-and-eval` status check. Decide deliberately which trusted maintainers may bypass it.
5. Keep Actions workflow permissions read-only by default. Grant write access only inside a job that requires it, such as CodeQL's `security-events: write` permission.
6. Confirm Dependabot alerts, security updates, and the weekly dependency-update configuration are active.
7. Set topics such as `education`, `family`, `local-first`, `slack`, and `ai`.

Do not require a status-check name until GitHub has observed its first successful run. Otherwise a typo or unseen check can make the ruleset impossible to satisfy.

## 5. Verify the public boundary

After the push, inspect the repository as a logged-out visitor or in a private browser window. Confirm:

- the README starts with the new-family path;
- the MIT License, security policy, contribution guide, and issue forms render;
- Actions show passing behavioral and CodeQL workflows;
- repository search finds no names, email addresses, Slack IDs, provider keys, purchased-resource references, local paths, or coding-agent review notes;
- cloning into a new directory and following the documented installation path works without access to the maintainer's machine.

If private data or a credential appears in Git history, stop publication work, rotate the credential when applicable, and follow a deliberate history-rewrite and disclosure process. Deleting only the current file is not sufficient.

## 6. Tag only after release gates are met

The initial public `main` branch may remain an alpha snapshot. Create a `v0.1.0` tag and GitHub release only after the intended release commit passes hosted CI, a clean-clone walkthrough, the required provider probes, and deliberate operator approval. Family-specific answer reviews remain local evidence and must never be attached to the release.
