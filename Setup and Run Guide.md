# ClinicalSignBridge: Setup and Run Guide

> **Current Docker-based MVP**\
> **Local demonstration only**

## 1. Understand what needs to run

The application currently consists of five Docker Compose services.

  -----------------------------------------------------------------------
  Component               Technology              Responsibility
  ----------------------- ----------------------- -----------------------
  Patient app             Angular                 Video upload, candidate
                                                  review, confirmation
                                                  and correction

  Doctor app              Angular                 Consultation and
                                                  patient communication
                                                  history

  Backend                 Python, FastAPI         Business logic,
                                                  sessions, recognition
                                                  orchestration,
                                                  messaging

  AI engine               Python, PyTorch,        ISL video inference
                          MediaPipe               

  Database                PostgreSQL 17           Consultations,
                                                  recognitions, messages,
                                                  confirmations and audit
                                                  records
  -----------------------------------------------------------------------

The AI engine is accessed internally by the backend, not directly by the
browser.

## 2. Machine prerequisites

The other machine should have:

  -----------------------------------------------------------------------
  Prerequisite                        Requirement
  ----------------------------------- -----------------------------------
  OS                                  Windows 10/11 64-bit

  Docker                              Docker Desktop with WSL2 backend

  Git                                 Git for Windows

  PowerShell                          Windows PowerShell 5.1 or
                                      PowerShell 7

  RAM                                 16 GB recommended

  Disk                                Allow at least 15--20 GB free for
                                      images, dependencies and model
                                      artifacts

  Browser                             Current Chrome or Edge

  Network                             Internet access for initial
                                      repository clone and
                                      image/dependency downloads
  -----------------------------------------------------------------------

The application runs inside Docker, so the person does not need to
separately install Python, Node.js, Angular CLI, PostgreSQL or PyTorch
on their Windows machine.

**Important:** The 16 GB RAM and disk figures are practical
recommendations, not verified minimum requirements. Actual resource
consumption depends on the model and Docker configuration.

## 3. Get the source code

Create a working directory:

``` powershell
New-Item -ItemType Directory -Force C:\Code\Learning\Spark
Set-Location C:\Code\Learning\Spark
```

Clone the repository:

``` powershell
git clone <YOUR_REPOSITORY_URL> clinical-sign-bridge
Set-Location .\clinical-sign-bridge
```

Check that the repository contains the expected directories:

``` powershell
Get-ChildItem
```

Expected major directories:

``` text
clinical-sign-bridge/
├── ai-engine/
├── backend/
├── patient-app/
├── doctor-app/
├── docker-compose.yml
├── .env.example
└── README.md
```

The exact Compose and environment-template filenames should be confirmed
against the final repository before distributing this guide.

## 3.a Dataset used: INCLUDE-50 dataset
Download instructions
1. Open the official INCLUDE Zenodo dataset page : https://zenodo.org/records/4010759
2. Locate the dataset archive files under the Files section.
3. Download the category ZIP archives.
4. Extract the archives into a common directory.
5. Locate the Train_Test_Split folder included in the extracted dataset.
6. Use the INCLUDE-50 CSV files to identify the predefined training and testing videos for the 50-class subset.
The dataset is split into category-based archives, including:

| Category                 | Number of Archive Parts |
|--------------------------|--------------------------|
| Adjectives               | 8 |
| Animals                  | 2 |
| Clothes                  | 2 |
| Colours                  | 2 |
| Days and Time            | 3 |
| Electronics              | 2 |
| Greetings                | 2 |
| Home                     | 4 |
| Jobs                     | 1 |
| Means of Transportation  | 2 |
| People                   | 5 |
| Places                   | 4 |
| Pronouns                 | 2 |
| Seasons                  | 1 |
| Society                  | 3 |

The complete dataset download is approximately 56.8 GB, before extraction.

Practical recommendation: For a reproducible first setup, download all archive parts and use the supplied INCLUDE-50 split files. This avoids accidentally omitting signs that belong to different categories.


## 4. Provision the BiLSTM model artifacts

> **Critical setup requirement**

The model artifacts are not included in the Git repository because the
`.pt` and `.npz` files are Git-ignored.

The AI engine requires these files:

``` text
ai-engine/
└── artifacts/
    └── bilstm/
        ├── best_model.pt
        ├── normalization.npz
        └── labels.json
```

Obtain the three files from your approved artifact package or securely
transfer them from your development environment.

Verify:

``` powershell
Test-Path .\ai-engine\artifacts\bilstm\best_model.pt
Test-Path .\ai-engine\artifacts\bilstm\normalization.npz
Test-Path .\ai-engine\artifacts\bilstm\labels.json
```

All three should return `True`.

Do not substitute a different model checkpoint or label map. They must
belong to the same BiLSTM V1 model version.

Also, the INCLUDE-50 dataset itself is not required to run inference. It
was used during model development and training.

**MediaPipe and OpenCV:** These are AI-engine runtime dependencies used
for landmark extraction and video processing. They are installed inside
the AI Docker image; they do not need to be installed separately on the
Windows host. Confirm the exact MediaPipe version from the repository
dependency lockfile before distributing the setup package.

## 5. Configure environment variables

Create the local `.env` file from the example:

``` powershell
Copy-Item .env.example .env
```

Configure the required PostgreSQL values:

``` dotenv
POSTGRES_DB=clinical_sign_bridge
POSTGRES_USER=clinical_user
POSTGRES_PASSWORD=<GENERATE_A_STRONG_LOCAL_PASSWORD>
```

Review the Compose configuration for the remaining application settings.

Important settings include:

  -------------------------------------------------------------------------------------
  Setting                                           Purpose
  ------------------------------------------------- -----------------------------------
  `CLINICAL_DATABASE_URL`                           Backend database connection,
                                                    constructed by Compose

  `CLINICAL_VIDEO_PREDICTION_PROVIDER`              Must be `bilstm-v1`

  `CLINICAL_BILSTM_ARTIFACTS_PATH`                  Model artifact directory

  `CLINICAL_RECOGNITION_PROCESSING_LEASE_SECONDS`   Recognition processing lease,
                                                    default 180 seconds

  AI connection timeout                             2 seconds

  AI response timeout                               60 seconds
  -------------------------------------------------------------------------------------

Do not commit `.env` or share a real database password through email or
source control.

## 6. Build and start the application

Open PowerShell in the repository root.

First, validate the Docker Compose configuration:

``` powershell
docker compose --env-file .env -p clinical-sign-bridge-b4 config
```

If that succeeds, build and start all services:

``` powershell
docker compose --env-file .env -p clinical-sign-bridge-b4 up -d --build --wait --wait-timeout 180
```

This should:

1.  Build the backend image.
2.  Build the AI inference image.
3.  Build the patient application image.
4.  Build the doctor application image.
5.  Start PostgreSQL.
6.  Run backend database migrations.
7.  Start the API and AI engine.
8.  Start both Angular applications.

The API startup process runs Alembic migrations. A fresh database should
therefore be initialized automatically.

If the Compose configuration does not successfully wait for all services
because of dependency or health-check behavior, inspect the individual
service status rather than repeatedly restarting everything.

## 7. Verify that all services are running

Execute:

``` powershell
docker compose --env-file .env -p clinical-sign-bridge-b4 ps
```

Expected services:

  Service     Expected state
  ----------- -----------------
  db          Healthy
  api         Healthy
  ai-engine   Running/healthy
  patient     Running
  doctor      Running

Check API readiness:

``` powershell
Invoke-RestMethod http://localhost:8000/ready
```

Check AI engine startup logs:

``` powershell
docker compose --env-file .env -p clinical-sign-bridge-b4 logs ai-engine --tail 100
```

**Important:** The AI `/ready` response currently reports the legacy
inference provider, not the video inference provider. Do not interpret
`deterministic-mock` in that response as proof that BiLSTM video
inference is not running. The actual video prediction path must be
verified separately.

## 8. Open the applications

  Application           URL
  --------------------- -----------------------------
  Patient application   http://localhost:4200/
  Doctor application    http://localhost:4201/
  Backend readiness     http://localhost:8000/ready

The AI engine uses internal port 8001 and is not directly exposed to the
Windows host in the current Compose setup.

## 9. Run an end-to-end demonstration

Use synthetic data only.

1.  **Create consultation:** Open the doctor application and create a
    new consultation.
2.  **Join as patient:** Open the patient application, enter the
    consultation code and accept the consent notice.
3.  **Upload a video:** Upload a supported MP4 or MOV video. For the
    current demo, use `MVI_4638.MOV` if you have been provided with that
    test file.
4.  **Review AI candidates:** The AI engine generates five candidate
    labels. These are unconfirmed isolated-sign predictions, not
    clinical interpretations.
5.  **Confirm or correct:** Select a candidate and explicitly confirm
    it, or correct the text.
6.  **Verify doctor communication:** Confirm that the patient-confirmed
    message appears in the doctor application and persists after
    refresh.

The current implementation accepts MP4 and MOV uploads. Camera
recordings are WebM and cannot currently be submitted through the
recognition workflow.

## 10. Stop and restart the application

Stop the containers without deleting persistent database data:

``` powershell
docker compose --env-file .env -p clinical-sign-bridge-b4 stop
```

Start them again:

``` powershell
docker compose --env-file .env -p clinical-sign-bridge-b4 start
```

Check status:

``` powershell
docker compose --env-file .env -p clinical-sign-bridge-b4 ps
```

Avoid this command unless you intentionally want to delete the local
database volume:

``` powershell
docker compose --env-file .env -p clinical-sign-bridge-b4 down -v
```

The `-v` option deletes persistent volumes, including the PostgreSQL
data.

------------------------------------------------------------------------

## Important limitations to communicate

This is where I would be particularly clear when handing the application
to another person.

  Area                                  Current status
  ------------------------------------- -------------------------------------------------
  Patient/doctor authentication         Not implemented
  Consultation access                   Temporary join code/session
  Real patient information              Not permitted for this MVP
  ISL continuous sentence translation   Not implemented
  Clinical phrase interpretation        Not implemented
  AI confidence calibration             Not implemented
  Camera WebM submission                Not supported
  Production deployment                 Not ready
  Model                                 BiLSTM V1, trained on INCLUDE-50 isolated signs

The application is a working technical prototype, not a clinically
validated ISL translation product.

## My recommendation before distributing it

I would not ask someone else to follow a collection of chat
instructions. Package this into a small, reproducible onboarding bundle.

### Suggested handover package

-   `README.md`: Quick start and prerequisites.
-   `SETUP-GUIDE.md`: Detailed Windows setup instructions.
-   `.env.example`: Safe environment template.
-   `MODEL-ARTIFACTS.md`: Artifact download location, version and
    checksums.
-   `TROUBLESHOOTING.md`: Docker, database, migration, AI and browser
    issues.
-   `START.ps1`: Validate prerequisites, configure and start services.
-   `STOP.ps1`: Stop services without deleting data.
-   `VERIFY.ps1`: Check service health and basic connectivity.

**One important architectural improvement:** Add a dedicated
video-inference readiness check that explicitly reports the loaded model
version and artifact validation status. The current generic AI readiness
response is misleading for someone diagnosing a fresh installation.
