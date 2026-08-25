# Deploying

## Render (free, recommended)

Render offers free Docker web services. `render.yaml` in this repo configures
everything.

1. Sign in at https://render.com with GitHub
2. **New +** -> **Blueprint**
3. Select this repository
4. Render reads `render.yaml` -> **Apply**

The first build takes about 10 minutes. You get one URL serving both the API
and the map.

### What the free tier means

| Constraint | Consequence |
|---|---|
| No persistent disk | Demo scenes and the incident registry are baked into the image at build time, so a cold container still has data. Scenes fetched at runtime are lost on restart. |
| Sleeps after 15 min idle | The next request takes ~50 s to wake the service. Open it once before a demo. |
| 512 MB RAM | Enough for the classical detector. Not enough for the U-Net, which is not deployed anyway. |

---

## Local, two tiers

```bash
docker compose up --build
# UI  http://localhost:8080
# API http://localhost:8000/docs
```

`SERVE_UI=false` turns the API into a pure API and `docker/Dockerfile.frontend`
serves the UI separately - the split matters on a paid plan where the two can
scale independently. On a free plan one service is better, because each
sleeping service is another cold start.

---

## Local, single process

```bash
python -m uvicorn api.main:app --port 8000
```

---

## After deploying

`GET /api/pipeline/capabilities` reports whether storage is writable, the real
imagery and AIS lag, and which sources are open versus token-gated.

Fetch fresh imagery (no account needed):

```bash
curl -X POST https://<your-app>/api/pipeline/fetch \
  -H 'Content-Type: application/json' \
  -d '{"bbox":[68,8,78,20],"days":3,"max_scenes":2}'
```

Returns a job id; poll `GET /api/pipeline/jobs/<id>`.

---

## Not Hugging Face Spaces

Hugging Face now requires a PRO subscription for Docker Spaces on cpu-basic;
only static Spaces remain free. Creating one returns HTTP 402.
