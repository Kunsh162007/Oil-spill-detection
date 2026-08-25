# Deploying

The application is two tiers. You can deploy the API alone (it can serve the
UI itself) or both separately.

---

## Option 1 — Hugging Face Spaces (recommended, genuinely free)

Free, Docker-native, **16 GB RAM**, and no credit card. The most capable free
option for an image this size (~3 GB).

1. Create a Space at https://huggingface.co/new-space
   - **SDK:** Docker
   - **Hardware:** CPU basic (free)
2. Clone it and copy this project in:

   ```bash
   git clone https://huggingface.co/spaces/<your-user>/<space-name>
   cd <space-name>
   # copy everything except the venv and data
   cp -r /path/to/Oil-spill-detection/{Dockerfile,requirements.txt,core,ingest,detect,drift,attribute,decision,api,ui,scripts,configs} .
   cp /path/to/Oil-spill-detection/SPACE_README.md README.md
   git add -A && git commit -m "deploy" && git push
   ```

3. The Space builds and starts on port 7860. First build takes ~10 minutes.

`SPACE_README.md` already carries the YAML front matter Spaces needs — it must
be named `README.md` in the Space.

---

## Option 2 — Render

Free web services work but have **no persistent disk**, so analysed scenes are
lost on each cold start and re-computed. Demo scenes are baked into the image,
so the app still works.

1. New → Web Service → connect the GitHub repo
2. Runtime **Docker**, health check path `/api/health`
3. Environment:
   - `SERVE_UI=true` (single service), or `false` if hosting the UI separately
   - `CORS_ORIGINS=https://<your-frontend>` when the UI is separate

`render.yaml` in this repo defines the full two-service topology with a
persistent disk, for when you move off the free tier.

---

## Option 3 — Two tiers locally

```bash
docker compose up --build
# UI  http://localhost:8080
# API http://localhost:8000/docs
```

---

## After deploying

Check `GET /api/pipeline/capabilities` — it reports whether storage is
writable, the real imagery and AIS lag, and which data sources are open versus
token-gated.

To pull fresh imagery (no account needed):

```bash
curl -X POST https://<your-app>/api/pipeline/fetch \
  -H 'Content-Type: application/json' \
  -d '{"bbox":[68,8,78,20],"days":3,"max_scenes":2}'
```

That returns a job id; poll `GET /api/pipeline/jobs/<id>`.

---

## Notes

- The image is CPU-only by design. The trained checkpoint is not deployed —
  see the README on why it is not yet validated at scene level.
- `.dockerignore` keeps the virtualenv and raw imagery out of the build
  context; without it the build ships gigabytes to the daemon.
- The build fails deliberately if a source file copies in empty, which is what
  a disk-full build produces otherwise.
