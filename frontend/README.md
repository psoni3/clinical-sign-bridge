# Angular patient and doctor applications

This workspace contains standalone Angular projects for the patient and doctor
applications. The patient app includes a separate MP4/MOV upload for unconfirmed
BiLSTM V1 video predictions. It does not confirm predictions or create messages.

From this directory:

```powershell
npm ci --legacy-peer-deps
npm run start -- --project patient --port 4200
npm run start -- --project doctor --port 4201
npm test
npm run build:patient
npm run build:doctor
```

`npm run lint` checks formatting with Prettier. TypeScript and Angular template checks
run in the production builds.

The patient app's AI prediction URL is configured in
`projects/patient/src/environments/environment.ts`; the development build replaces it
with `environment.development.ts`, which defaults to `http://localhost:8000/api/v1`.
Set the environment value to the deployed AI API base URL before creating a production
build.

The prediction service must be running with the BiLSTM V1 provider enabled. From the
repository root, install the AI engine's inference dependencies and start it on port 8000:

```powershell
uv sync --project ai-engine --extra preprocessing --extra training --group dev
$env:CLINICAL_VIDEO_PREDICTION_PROVIDER = "bilstm-v1"
$env:CLINICAL_CORS_ORIGINS = '["http://localhost:4200"]'
uv run --project ai-engine uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

In another terminal, start the patient app from this directory:

```powershell
npm run start -- --project patient --port 4200
```

`CLINICAL_CORS_ORIGINS` is a JSON array of explicit HTTP(S) origins. The AI API
rejects wildcard origins and permits only its configured methods and headers. Uploads
are limited in the browser for usability; server-side validation remains authoritative.
See the repository-root README for the complete setup and privacy boundary.
