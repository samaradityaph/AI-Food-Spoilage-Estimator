# Railway Deployment Guide

## Quick Deploy to Railway

### Step 1: Push to GitHub

```bash
git add .
git commit -m "Add Railway deployment configuration"
git push origin main
```

### Step 2: Deploy on Railway

1. Go to [railway.app](https://railway.app) and sign in with GitHub
2. Click **"New Project"** → **"Deploy from GitHub repo"**
3. Select your repository
4. Railway will automatically detect the `Dockerfile`

### Step 3: Configure Environment Variables

In Railway Dashboard → Your Project → **Variables** tab, add:

| Variable | Value | Required |
|----------|-------|----------|
| `HUGGINGFACE_API_TOKEN` | `hf_your_token_here` | ✅ Yes |
| `FLASK_ENV` | `production` | ✅ Yes |
| `ALLOWED_ORIGINS` | `https://your-app.up.railway.app` | Optional |

> **⚠️ IMPORTANT**: Get your Hugging Face token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### Step 4: Generate Domain

1. Go to **Settings** → **Networking**
2. Click **"Generate Domain"** to get a public URL
3. Your app will be live at `https://your-app.up.railway.app`

---

## Security Features Included

| Feature | Description |
|---------|-------------|
| **Non-root user** | Container runs as unprivileged user |
| **Gunicorn WSGI** | Production server (not Flask dev server) |
| **Rate limiting** | 60 requests/minute per IP |
| **Security headers** | XSS, clickjacking, CSP protection |
| **CORS restriction** | Configurable allowed origins |
| **Health checks** | Automatic container health monitoring |
| **10MB upload limit** | Prevents abuse |

---

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `HUGGINGFACE_API_TOKEN` | - | **Required**. Your HF API token |
| `FLASK_ENV` | `production` | Set to `development` for local testing |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins |
| `RATE_LIMIT_REQUESTS` | `60` | Max requests per window |
| `RATE_LIMIT_WINDOW` | `60` | Window size in seconds |
| `MAX_CONTENT_LENGTH` | `10485760` | Max upload size (bytes) |
| `PORT` | `5000` | Server port (Railway sets this automatically) |

---

## Monitoring

### View Logs
```bash
railway logs
```

### Check Health
Your app's health endpoint: `https://your-app.up.railway.app/api/foods`

---

## Costs

Railway offers:
- **Free Tier**: $5 credit/month (enough for ~500 hours)
- **Pro**: $5/month + usage

For a low-traffic app, the free tier should be sufficient.
