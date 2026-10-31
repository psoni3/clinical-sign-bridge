# Angular application shells

This workspace contains standalone Angular projects for the patient and doctor shells.
Neither project implements consultation or recognition workflows yet.

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
run in the production builds. See the repository-root README for the complete Compose
setup, privacy boundary, and backend/inference commands.
