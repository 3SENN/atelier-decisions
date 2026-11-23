# Security policy

This repository makes the decision models Atelier runs on your computer: atelier-nano, built in, and atelier-mini,
an optional download. Atelier reads their answers to decide small things for you, some of them about safety (is
this sender who they say they are, does this text expose a secret). We take reports about them seriously.

## Reporting a vulnerability

Please report security issues privately through GitHub's
[private vulnerability reporting](https://github.com/3SENN/atelier-decisions/security/advisories/new)
(the **Report a vulnerability** button on the repository's Security tab).
If you can't use GitHub, email [contact@chinnovations.nl](mailto:contact@chinnovations.nl) instead.
Do not open a public issue, pull request or discussion for a security problem.

Report it this way when:

- a runner (`runtime/node`, `student/serve.py`, `student/serve_mini.py`, `teacher/server.py`) can be made to do
  something it shouldn't: answer beyond 127.0.0.1, read or write files outside its folders, run code from a request;
- a model file here or on [Hugging Face](https://huggingface.co/3SENN/atelier-decisions) could run code when it is
  loaded, or doesn't match the SHA-256 in its version's `MANIFEST.json`;
- Atelier's download manifest for the mini (`tools/mini_dist.py`, `miniManifest.ts`) can be made to accept a file it
  shouldn't;
- text an attacker controls (a page, an email, a file) reliably makes a model give a sure, wrong answer to one of
  Atelier's safety questions, beyond the limits its model card already states.

A good report includes what an attacker can do and what they need first, the version (`releases/<version>`), the
steps or the request that shows it, and any idea you have for a fix.

**Not a vulnerability:** a model being wrong now and then, including at the sure-and-wrong rates its model card
states. When a model isn't sure, Atelier's rule decides, and the models are never Atelier's only guard. Please open a
**Wrong decision** issue for those. Problems in Atelier itself go to
[Atelier's security policy](https://github.com/3SENN/atelier/blob/main/SECURITY.md).

## What happens next

- **Within 3 business days** we acknowledge your report.
- We confirm the problem, keep you posted while we work on it, and **aim to ship a fix within 90 days**; sooner for
  anything under active attack.
- We publish a **GitHub security advisory** for every fixed vulnerability, with a CVE requested through GitHub, and
  agree on the disclosure date with you.
- We credit you in the advisory unless you prefer not to be named.

## Supported versions

Security fixes go into the newest saved version and the code on `main`. Saved versions never change (each is
hashed in its `MANIFEST.json`), so a fix to a model is a new version.

## Safe harbour

We will not take legal action against anyone who reports in good faith, avoids harming other people's data and
privacy, and gives us reasonable time to fix the issue before disclosing it.
