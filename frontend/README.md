# Angular patient and doctor applications

This workspace contains standalone Angular projects for the patient and doctor
applications. Both applications communicate with the backend API. Patient MP4/MOV
recognition uploads go to the consultation recognition endpoint; persisted candidates
are retrieved from the backend before the patient reviews or confirms them.

The backend communicates with the AI inference service over the internal service
network. In Docker Compose, the backend uses `http://ai-engine:8001`; the AI service
port is not published to the host or exposed to the browser. Do not configure an
Angular URL for the AI inference service.

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

The frontend API clients default to `http://localhost:8000/api/v1`. Start the backend
and its configured internal AI inference service using the repository-root setup
instructions before testing recognition uploads. The patient file picker accepts
MP4 and MOV only; the camera still records WebM, which the backend does not accept.

Start the Angular applications from this directory:

```powershell
npm run start -- --project patient --port 4200
npm run start -- --project doctor --port 4201
```

See the repository-root README for backend, AI artifact, and local deployment setup.
