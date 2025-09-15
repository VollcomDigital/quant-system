# Security Policy

## Supported Versions

We aim to keep the `main` branch secure and up-to-date. Security updates are provided on a best-effort basis for the latest release.

## Reporting a Vulnerability

Please report security issues privately. Do not create public GitHub issues for vulnerabilities.

- Email: [36189959+LouisLetcher@users.noreply.github.com](mailto:36189959+LouisLetcher@users.noreply.github.com)
- Or use GitHub’s “Report a vulnerability” (Security > Advisories) if available

We will acknowledge your report within 72 hours, provide an initial assessment, and keep you informed of the remediation progress.

## Best Practices

- Do not include secrets in commits. Use `.env` files and repository secrets.
- Use Dockerized workflows or Poetry to ensure reproducible environments.
- Keep dependencies current; Dependabot is configured for this repository.
