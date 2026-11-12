# Developer Tools and Access

This guide covers access to source control, cloud environments and engineering tools. Request access in the Self Service portal unless stated otherwise.

## Getting GitHub access

New engineers are added to the company GitHub organization on their first day. If you cannot see the company repositories, check that you signed in with your company SSO account rather than a personal one. Enable two-factor authentication on GitHub before access is granted. Access to a specific repository is requested through the Self Service portal and approved by the repository owner, usually within 1 business day.

## SSH keys and commit signing

Generate an ed25519 key with `ssh-keygen -t ed25519`, add the public key under GitHub settings, and authorize it for SSO. Keys older than 12 months are flagged for rotation. All commits to protected branches must be signed. If you get a "Permission denied (publickey)" error, run `ssh -T git@github.com` to check which key is offered, and confirm the key is authorized for the organization.

## Cloud and production access

Production access is never permanent. Request just-in-time access through the access portal, state the ticket that justifies it, and choose a duration of up to 8 hours. Approval needs your manager and the service owner. Every production session is recorded. Use the staging environment for testing; staging data is anonymized and may be used freely.

## API keys and secrets

Never commit secrets or API keys to a repository. Store them in the company secrets manager and read them at runtime. If a key is committed by mistake, rotate it immediately and report it to security@company.com within 1 hour, even if the repository is private. Automated scanning blocks pushes that contain known key patterns.

## Running the VPN with developer tools

Internal package registries and staging systems are only reachable when connected to the VPN. If package installs fail with timeouts or certificate errors, connect to the VPN, then set the registry URL from the engineering handbook. Do not disable certificate verification to work around errors.

## CI pipeline failures

When a build fails, open the pipeline log and find the first failing step instead of the last. Flaky tests can be re-run once from the pipeline page. If the same test fails repeatedly on the main branch, post in the engineering channel and open a ticket with the category Developer Tools. Build runners are restarted every night at 2 a.m. Eastern time.
