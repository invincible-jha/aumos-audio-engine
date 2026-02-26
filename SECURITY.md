# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

**Do not open public GitHub issues for security vulnerabilities.**

Report security issues to: security@muveraai.com

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact assessment
- Suggested fix (if available)

We will respond within 48 hours and aim to patch within 14 days.

## Security Considerations for Audio Engine

### Biometric Data Protection

This service processes voice audio which may contain biometric identifiers.

**Critical requirements:**
- Speaker de-identification is applied before any audio is stored persistently
- No raw voice biometrics are retained beyond the processing pipeline
- All intermediate audio files are encrypted at rest and deleted after processing
- Privacy engine validation is required before synthesis jobs complete

### MNPI Detection

The MNPI (Material Non-Public Information) detection system is a compliance aid, not
a legal guarantee. False negatives are possible. Always combine with human review
for regulated financial communications.

### Tenant Isolation

- All audio jobs are scoped to a tenant via RLS
- Cross-tenant audio access is blocked at the database level
- Storage paths are namespaced by tenant ID

### Data Retention

- Synthesis job metadata: retained per tenant data retention policy
- Processed audio files: deleted after configurable retention window
- Transcription results: treated as sensitive financial data

### Authentication

All API endpoints require valid JWT tokens issued by aumos-auth-gateway.
No unauthenticated access is permitted.
