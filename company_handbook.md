# Acme Cloud, Company Handbook & Product FAQ (sample RAG corpus)

This is synthetic sample data for the RAG sessions and the email/article projects.

## About Acme Cloud
Acme Cloud is a (fictional) platform that provides managed Postgres, object storage,
and serverless functions. Founded in 2019, headquartered in Riyadh, Saudi Arabia, with a
remote-first team across the MENA region.

## Plans & Pricing
- **Starter**, $0/month. 1 project, 1 GB Postgres, 5 GB storage, community support.
- **Pro**, $29/month. 10 projects, 50 GB Postgres, 250 GB storage, email support, daily backups.
- **Scale**, $199/month. Unlimited projects, 500 GB Postgres, 2 TB storage, priority support, point-in-time recovery (PITR), SSO.
- Overage storage is billed at $0.02/GB. Egress is free up to 100 GB/month.

## Support & SLA
- Pro plan: email support, first response within 24 business hours.
- Scale plan: priority support, first response within 4 business hours, 99.95% uptime SLA.
- Status page: status.acme.example. Incidents are posted within 15 minutes of detection.

## Backups & Recovery
- Pro: automatic daily backups retained for 7 days.
- Scale: PITR with 1-second granularity, 35-day retention.
- Restores can be triggered from the dashboard or via the CLI `acme db restore`.

## Refund Policy
- Monthly plans can be cancelled anytime; service runs until the end of the billing period.
- Annual plans: prorated refund within the first 30 days, no refund afterward.
- To request a refund, email billing@acme.example with your account ID.

## Security
- Data encrypted at rest (AES-256) and in transit (TLS 1.3).
- SOC 2 Type II certified. SSO (SAML/OIDC) available on Scale.
- Customers can request a DPA and the latest pen-test summary from security@acme.example.

## Common Support Answers
- **Reset password:** dashboard → Account → Security → "Send reset link".
- **Increase connection limit:** Pro allows 60 connections; Scale allows 500. Use a pooler (PgBouncer) for more.
- **Region availability:** eu-central, us-east, and me-central (Riyadh).
- **Data export:** `acme db dump` produces a standard `pg_dump`; storage export via the CLI `acme storage sync`.
