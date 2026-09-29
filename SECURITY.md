# Security

This is a local research/learning application. The Flask viewer is a development
server, binds to localhost by default, and is not designed as a public hosted service.
Keep it local unless authentication, input validation and deployment hardening are added.

No API key or external model service is required. Neural inference and training run
locally. Installing dependencies requires network access; the viewer currently requests
fonts from Google Fonts.

Only load checkpoints from sources you trust. Neural loading uses PyTorch's
`weights_only=True`. Local models, experiment runs, credentials, environments, and
logs should remain excluded from version control.

To check a change before publishing, install Gitleaks and run:

```sh
gitleaks git . --log-opts=--all --redact
```

Also scan the proposed release files, because a history scan alone does not cover
uncommitted files. A clean scan is evidence, not a guarantee that all sensitive
information has been detected.

Report vulnerabilities through this repository's private vulnerability reporting
feature when available. Otherwise open an issue requesting a private contact channel,
without including exploit details, credentials, or personal data.
