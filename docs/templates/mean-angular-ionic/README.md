# Angular/Ionic MEAN reference templates

This is a reference-only template (not wired to the app) for a hyper‑modular Angular/Ionic client and a Node/Express server with MEAN conventions.

Key rules:
- No client .env. Secrets live only in server `.env` (ignored). Check in `server/.env.example` only.
- Client is feature-first and hyper‑modular: `client/src/app/features/<feature>` with lazy routes, standalone components.
- API flow: Page → Components → Feature Service → ApiService → Server.
- Loader/Resolver fetches current user; Guard enforces permissions per tool.
- Stub→Real: Use a Port with a Stub (mock data) and a Real service (ApiService). Switch via `environment.useStub` or a cloud flag endpoint.
- CI/CD assumes `client/` and `server/` roots in this template; Dockerfiles under `shared/docker/`.

See `client/`, `server/`, `shared/docker/`, and `.github/workflows/` for examples.


