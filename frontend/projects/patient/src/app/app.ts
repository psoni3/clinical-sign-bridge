import { HttpErrorResponse } from '@angular/common/http';
import {
  ChangeDetectorRef,
  Component,
  ElementRef,
  OnDestroy,
  ViewChild,
  inject,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { firstValueFrom } from 'rxjs';

import {
  CandidateDto,
  InterpretationDto,
  MessageDto,
  PatientApiService,
  RecognitionDto,
} from './patient-api.service';

type PatientStage = 'join' | 'consent' | 'setup' | 'capture' | 'review' | 'conversation';

const MAX_RECORDING_SECONDS = 30;
const MAX_UPLOAD_BYTES = 50 * 1024 * 1024;
const RECOGNITION_POLL_LIMIT = 12;
const RECOGNITION_POLL_INTERVAL_MS = 1000;
const SESSION_RECOVERY_ERROR_CODES = new Set([
  'SESSION_NOT_FOUND',
  'SESSION_EXPIRED',
  'SESSION_REVOKED',
  'SESSION_MISMATCH',
  'PATIENT_SESSION_REQUIRED',
]);

function isPendingRecognition(status: RecognitionDto['status']): boolean {
  return status === 'QUEUED' || status === 'PROCESSING';
}

function apiErrorCode(error: unknown): string | null {
  if (!(error instanceof HttpErrorResponse)) {
    return null;
  }
  const payload: unknown = error.error;
  if (
    typeof payload !== 'object' ||
    payload === null ||
    !('detail' in payload) ||
    typeof payload.detail !== 'object' ||
    payload.detail === null ||
    !('code' in payload.detail) ||
    typeof payload.detail.code !== 'string'
  ) {
    return null;
  }
  return payload.detail.code;
}

function isSessionRecoveryError(error: unknown): boolean {
  const code = apiErrorCode(error);
  return code !== null && SESSION_RECOVERY_ERROR_CODES.has(code);
}

function validateRecognitionFile(file: Pick<File, 'name' | 'type' | 'size'>): string | null {
  const extension = file.name.split('.').pop()?.toLowerCase();
  const expectedMimeType =
    extension === 'mp4' ? 'video/mp4' : extension === 'mov' ? 'video/quicktime' : null;
  if (!extension || !expectedMimeType) {
    return 'Choose a video with an MP4 or MOV file extension.';
  }
  if (file.type.toLowerCase() !== expectedMimeType) {
    return `The selected ${extension.toUpperCase()} file must have MIME type ${expectedMimeType}.`;
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    return 'This video exceeds the 50 MiB upload limit.';
  }
  return null;
}

function errorMessage(error: unknown): string {
  if (error instanceof HttpErrorResponse) {
    const code = apiErrorCode(error);
    if (code && SESSION_RECOVERY_ERROR_CODES.has(code)) {
      return 'Your temporary patient session is no longer valid. Rejoin with the consultation ID and join code; the old session will not be reused.';
    }
    if (code === 'CONSULTATION_EXPIRED') {
      return 'This consultation has expired. Ask the clinician to start a new consultation.';
    }
    if (code === 'INVALID_JOIN_CODE') {
      return 'The join code was not accepted. Check the consultation ID and join code with the clinician.';
    }
    const payload: unknown = error.error;
    if (typeof payload === 'object' && payload !== null) {
      if ('message' in payload && typeof payload.message === 'string') {
        return payload.message;
      }
      if ('detail' in payload && typeof payload.detail === 'object' && payload.detail !== null) {
        const detail = payload.detail;
        if ('message' in detail && typeof detail.message === 'string') {
          return detail.message;
        }
      }
    }
    if (error.status === 0) {
      return 'The service could not be reached. Check the connection and try again.';
    }
  }
  return 'The request could not be completed. Please try again.';
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [FormsModule],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App implements OnDestroy {
  private readonly api = inject(PatientApiService);
  private readonly changeDetector = inject(ChangeDetectorRef);
  private recorder: MediaRecorder | null = null;
  private recordingChunks: BlobPart[] = [];
  private recordingStartedAt = 0;
  private recordingTimer: ReturnType<typeof setInterval> | null = null;
  private recognitionFileRequestId: string | null = null;
  private pendingRecognition: { clientRequestId: string; recognitionId: string } | null = null;
  private socket: WebSocket | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectDelay = 1000;
  private clipUrl: string | null = null;
  private confirmationRequest: { candidateId: string; key: string } | null = null;
  private correctionRequest: { candidateId: string; text: string; key: string } | null = null;
  private destroyed = false;

  @ViewChild('preview')
  set preview(element: ElementRef<HTMLVideoElement> | undefined) {
    if (element && this.cameraStream) {
      element.nativeElement.srcObject = this.cameraStream;
    }
  }

  @ViewChild('recognitionVideoInput')
  private recognitionVideoInput?: ElementRef<HTMLInputElement>;

  stage: PatientStage = 'join';
  consultationId = '';
  joinCode = '';
  sessionId = '';
  conversationId = '';
  error = '';
  statusMessage = '';
  busy = false;
  consentAccepted = false;
  cameraStream: MediaStream | null = null;
  cameraAvailable = false;
  cameraStarting = false;
  liveConsentRecorded = false;
  cameraError = '';
  isRecording = false;
  recordingSeconds = 0;
  recordedClip: Blob | null = null;
  interpretation: InterpretationDto | null = null;
  selectedCandidateId: string | null = null;
  recognitionFile: File | null = null;
  recognitionFileError = '';
  recognitionFileStatus = '';
  correctionText = '';
  messages: MessageDto[] = [];
  messageText = '';
  confirmedText = '';
  socketConnected = false;
  consultationClosed = false;
  readonly visualResponses = ['Yes', 'No', 'Not sure', 'Repeat question', 'I do not understand'];

  get selectedCandidate(): CandidateDto | undefined {
    return this.interpretation?.candidates.find(
      (candidate) => candidate.candidateId === this.selectedCandidateId,
    );
  }

  get secondsRemaining(): number {
    return Math.max(0, MAX_RECORDING_SECONDS - this.recordingSeconds);
  }

  get hasDoctorQuestions(): boolean {
    return this.messages.some((message) => message.senderType === 'DOCTOR');
  }

  get connectionStatus(): string {
    if (this.consultationClosed) {
      return 'Consultation ended. Conversation is read-only.';
    }
    if (this.socketConnected) {
      return 'Live updates connected.';
    }
    return 'Reconnecting to live updates; messages reload from the service.';
  }

  get recordedPreviewUrl(): string | null {
    return this.clipUrl;
  }

  get hasPendingRecognition(): boolean {
    return (
      this.recognitionFileRequestId !== null &&
      this.pendingRecognition?.clientRequestId === this.recognitionFileRequestId
    );
  }

  get recognitionSubmitLabel(): string {
    if (this.busy) {
      return 'Processing…';
    }
    if (this.hasPendingRecognition) {
      return 'Check recognition status';
    }
    return 'Submit video';
  }

  onRecognitionFileSelected(event: Event): void {
    const input = event.target;
    if (!(input instanceof HTMLInputElement)) {
      return;
    }

    const previousFile = this.recognitionFile;
    const previousRequestId = this.recognitionFileRequestId;
    this.recognitionFile = null;
    this.recognitionFileRequestId = null;
    this.recognitionFileError = '';
    this.recognitionFileStatus = '';

    const file = input.files?.[0];
    if (!file) {
      this.pendingRecognition = null;
      return;
    }

    const validationError = validateRecognitionFile(file);
    if (validationError) {
      this.pendingRecognition = null;
      this.recognitionFileError = validationError;
      input.value = '';
      return;
    }

    this.recognitionFile = file;
    if (previousFile === file && previousRequestId) {
      this.recognitionFileRequestId = previousRequestId;
    } else {
      this.pendingRecognition = null;
      this.recognitionFileRequestId = crypto.randomUUID();
    }
    this.recognitionFileStatus =
      'Video selected. Upload starts only after you choose Submit video.';
    this.changeDetector.markForCheck();
  }

  async uploadRecognitionFile(): Promise<void> {
    if (this.busy) {
      return;
    }
    const file = this.recognitionFile;
    if (!file) {
      this.recognitionFileError = 'Choose an MP4 or MOV file before submitting.';
      return;
    }
    if (!this.consultationId || !this.sessionId || !this.liveConsentRecorded) {
      this.recognitionFileError =
        'Join a consultation and record live-processing consent before submitting a video.';
      return;
    }
    if (this.consultationClosed) {
      this.recognitionFileError = 'This consultation is read-only. You cannot submit a video.';
      return;
    }
    const clientRequestId = (this.recognitionFileRequestId ??= crypto.randomUUID());
    const recognitionIdToPoll = this.hasPendingRecognition
      ? this.pendingRecognition?.recognitionId
      : undefined;
    this.recognitionFileStatus = recognitionIdToPoll
      ? 'Checking the existing recognition request.'
      : 'Submitting the selected file to the consultation backend.';
    const loaded = await this.processRecognitionUpload(
      file,
      clientRequestId,
      file.name,
      recognitionIdToPoll,
    );
    if (loaded) {
      this.recognitionFileStatus = 'Persisted candidates are ready for review above.';
    } else if (this.hasPendingRecognition) {
      this.recognitionFileStatus =
        'This request is still available on this page. Check its status later without uploading again.';
    } else if (this.error) {
      this.recognitionFileStatus = this.error;
    }
  }

  async join(): Promise<void> {
    this.error = '';
    this.statusMessage = '';
    if (!this.consultationId.trim() || !this.joinCode.trim()) {
      this.error = 'Enter both the consultation ID and the join code provided by the clinician.';
      return;
    }

    this.busy = true;
    try {
      const joined = await firstValueFrom(
        this.api.joinConsultation(this.consultationId.trim(), this.joinCode.trim()),
      );
      const consultation = await firstValueFrom(this.api.getConsultation(joined.consultationId));
      this.consultationId = joined.consultationId;
      this.sessionId = joined.sessionId;
      this.conversationId = consultation.conversationId;
      this.consultationClosed = false;
      this.stage = 'consent';
      this.statusMessage =
        'Consultation joined. Review the privacy notice and choose whether to continue.';
    } catch (error: unknown) {
      this.handleRequestError(error);
    } finally {
      this.busy = false;
      this.changeDetector.markForCheck();
    }
  }

  async acceptLiveConsent(): Promise<void> {
    this.error = '';
    if (!this.consentAccepted) {
      this.error = 'Select the consent checkbox before continuing to camera setup.';
      return;
    }
    this.busy = true;
    try {
      await firstValueFrom(this.api.recordLiveConsent(this.consultationId, this.sessionId, true));
      this.liveConsentRecorded = true;
      this.stage = 'setup';
      this.statusMessage = 'Video-processing consent recorded for this consultation.';
      this.connectNotifications();
      await this.refreshMessages();
    } catch (error: unknown) {
      this.handleRequestError(error);
    } finally {
      this.busy = false;
      this.changeDetector.markForCheck();
    }
  }

  async startCamera(): Promise<void> {
    if (this.cameraStarting) {
      return;
    }
    this.error = '';
    this.cameraError = '';
    if (!this.liveConsentRecorded) {
      this.stage = 'consent';
      this.cameraError = 'Accept video-processing consent before enabling the camera.';
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      this.cameraError =
        'Camera access is not available in this browser. You can still send a typed message.';
      return;
    }
    this.cameraStarting = true;
    this.stopCamera();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: 'user',
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 24, max: 30 },
        },
      });
      if (this.destroyed || !this.liveConsentRecorded) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      this.cameraStream = stream;
      this.cameraAvailable = this.cameraStream.getVideoTracks().length > 0;
      this.statusMessage = this.cameraAvailable
        ? 'Camera is ready. Framing guidance is provided, but this prototype does not assess image quality.'
        : '';
      if (!this.cameraAvailable) {
        this.cameraError =
          'No camera track was available. Retry camera access or send a typed message.';
        this.stopCamera();
      }
    } catch (error: unknown) {
      this.cameraAvailable = false;
      this.cameraError = this.cameraPermissionMessage(error);
    } finally {
      this.cameraStarting = false;
      this.changeDetector.markForCheck();
    }
  }

  continueToCapture(): void {
    if (!this.cameraAvailable) {
      this.cameraError = 'Connect a camera before continuing, or choose the typed-message option.';
      return;
    }
    this.stage = 'capture';
    this.statusMessage = 'Camera ready. Begin when you are comfortable.';
    this.error = '';
  }

  startRecording(): void {
    this.error = '';
    if (!this.cameraStream || !this.cameraAvailable) {
      this.cameraError = 'Camera is not available. Retry setup or send a typed message.';
      return;
    }
    const mimeType = ['video/webm;codecs=vp8', 'video/webm'].find((type) =>
      MediaRecorder.isTypeSupported(type),
    );
    if (!mimeType) {
      this.error =
        'This browser cannot create a supported video segment. Use the typed-message alternative.';
      return;
    }
    try {
      this.releaseRecordedClip();
      this.recordingChunks = [];
      this.recorder = new MediaRecorder(this.cameraStream, {
        mimeType,
        videoBitsPerSecond: 1_000_000,
      });
      this.recorder.ondataavailable = (event: BlobEvent) => {
        if (event.data.size > 0) {
          this.recordingChunks.push(event.data);
        }
      };
      this.recorder.onstop = () => this.finishRecording();
      this.recorder.start(250);
      this.recordingStartedAt = Date.now();
      this.recordingSeconds = 0;
      this.isRecording = true;
      this.recordingTimer = setInterval(() => {
        this.recordingSeconds = Math.floor((Date.now() - this.recordingStartedAt) / 1000);
        if (this.recordingSeconds >= MAX_RECORDING_SECONDS) {
          this.stopRecording();
        }
        this.changeDetector.markForCheck();
      }, 250);
    } catch {
      this.error = 'The camera could not start recording. Retry or send a typed message.';
    }
  }

  stopRecording(): void {
    if (this.recordingTimer !== null) {
      clearInterval(this.recordingTimer);
      this.recordingTimer = null;
    }
    if (this.recorder?.state === 'recording') {
      this.recorder.stop();
    }
    this.isRecording = false;
    this.changeDetector.markForCheck();
  }

  private finishRecording(): void {
    const clip = new Blob(this.recordingChunks, { type: this.recorder?.mimeType || 'video/webm' });
    this.recordingChunks = [];
    this.recorder = null;
    if (clip.size === 0) {
      this.error = 'No video was captured. Try recording again.';
      this.changeDetector.markForCheck();
      return;
    }
    if (clip.size > MAX_UPLOAD_BYTES) {
      this.error =
        'This recording is too large to keep for preview. Record a shorter segment or use typed text.';
      this.changeDetector.markForCheck();
      return;
    }
    this.recordedClip = clip;
    this.clipUrl = URL.createObjectURL(clip);
    this.statusMessage = `Camera recording ready to preview (${this.recordingSeconds} seconds). It is WebM and cannot be submitted for sign recognition yet.`;
    this.changeDetector.markForCheck();
  }

  async uploadSegment(): Promise<void> {
    if (!this.recordedClip) {
      this.error = 'Record a segment before uploading.';
      return;
    }
    this.error =
      'Camera recordings are WebM and cannot be submitted for recognition. Choose an MP4/MOV file or type a message instead.';
    this.statusMessage = '';
    this.changeDetector.markForCheck();
  }

  chooseRecognitionFile(): void {
    this.recognitionVideoInput?.nativeElement.click();
  }

  private async processRecognitionUpload(
    video: Blob,
    clientRequestId: string,
    filename?: string,
    recognitionIdToPoll?: string,
  ): Promise<boolean> {
    if (this.busy) {
      return false;
    }
    this.busy = true;
    this.error = '';
    this.statusMessage = recognitionIdToPoll
      ? 'Checking the existing recognition status...'
      : 'Uploading the video for recognition...';
    try {
      let recognition: RecognitionDto;
      if (recognitionIdToPoll) {
        try {
          recognition = await firstValueFrom(
            this.api.getRecognition(recognitionIdToPoll, this.sessionId),
          );
        } catch (error: unknown) {
          this.handleRecognitionPollingError(error);
          return false;
        }
      } else {
        recognition = await firstValueFrom(
          this.api.uploadSegment(
            this.consultationId,
            this.sessionId,
            clientRequestId,
            video,
            filename,
          ),
        );
        this.pendingRecognition = {
          clientRequestId,
          recognitionId: recognition.recognitionId,
        };
      }
      for (let attempt = 0; isPendingRecognition(recognition.status); attempt += 1) {
        if (attempt >= RECOGNITION_POLL_LIMIT) {
          this.error =
            'Recognition is still processing. Choose Check recognition status later on this page to resume the existing request without another upload. Refreshing loses this temporary session, so rejoin before continuing.';
          this.statusMessage = '';
          return false;
        }
        await new Promise((resolve) => setTimeout(resolve, RECOGNITION_POLL_INTERVAL_MS));
        try {
          recognition = await firstValueFrom(
            this.api.getRecognition(recognition.recognitionId, this.sessionId),
          );
        } catch (error: unknown) {
          this.handleRecognitionPollingError(error);
          return false;
        }
        this.statusMessage = `Recognition status: ${recognition.status.toLowerCase()}.`;
      }
      if (!recognition.interpretationId) {
        this.pendingRecognition = null;
        this.error =
          recognition.status === 'FAILED'
            ? 'Recognition failed. Your selected file and request ID are retained; submit again to retry.'
            : 'No interpretation was returned. Use typed text or request an interpreter.';
        return false;
      }

      try {
        this.interpretation = await firstValueFrom(
          this.api.getInterpretation(recognition.interpretationId, this.sessionId),
        );
      } catch (error: unknown) {
        if (isSessionRecoveryError(error)) {
          this.handleRequestError(error);
        } else {
          this.error = `Recognition completed, but its interpretation could not be retrieved. Choose Check recognition status later to retry retrieval without uploading again. ${errorMessage(error)}`;
        }
        return false;
      }
      this.pendingRecognition = null;
      this.selectedCandidateId = null;
      this.correctionText = '';
      this.stage = 'review';
      this.statusMessage =
        recognition.status === 'UNSUPPORTED'
          ? 'The mock provider cannot recognize this segment. No text was guessed or generated.'
          : 'Review the unverified interpretation before choosing an action.';
      return true;
    } catch (error: unknown) {
      if (isSessionRecoveryError(error)) {
        this.handleRequestError(error);
      } else if (this.pendingRecognition?.clientRequestId === clientRequestId) {
        this.handleRecognitionPollingError(error);
      } else {
        this.error = `${errorMessage(error)} The selected file and request ID remain on this page; retrying it will reuse the same request.`;
      }
      if (!isSessionRecoveryError(error)) {
        this.statusMessage = '';
      }
      return false;
    } finally {
      this.busy = false;
      this.changeDetector.markForCheck();
    }
  }

  async confirmSelectedCandidate(): Promise<void> {
    const candidate = this.selectedCandidate;
    const interpretation = this.interpretation;
    if (!candidate || !interpretation || interpretation.quality.status !== 'ACCEPTABLE') {
      this.error = 'Choose a candidate that is eligible for patient confirmation.';
      return;
    }
    await this.performInterpretationAction(async () => {
      if (this.confirmationRequest?.candidateId !== candidate.candidateId) {
        this.confirmationRequest = {
          candidateId: candidate.candidateId,
          key: crypto.randomUUID(),
        };
      }
      await firstValueFrom(
        this.api.confirmInterpretation(
          interpretation.interpretationId,
          candidate.candidateId,
          this.sessionId,
          this.confirmationRequest.key,
        ),
      );
      this.confirmationRequest = null;
      this.confirmedText = candidate.text;
      this.statusMessage =
        'Patient-confirmed. The exact selected text was sent to the conversation.';
      await this.refreshMessages();
      this.stage = 'conversation';
    });
  }

  async correctSelectedCandidate(): Promise<void> {
    const candidate = this.selectedCandidate;
    const interpretation = this.interpretation;
    const correctedText = this.correctionText.trim();
    if (!candidate || !interpretation || !correctedText) {
      this.error = 'Choose a candidate and enter your correction.';
      return;
    }
    await this.performInterpretationAction(async () => {
      if (
        this.correctionRequest?.candidateId !== candidate.candidateId ||
        this.correctionRequest.text !== correctedText
      ) {
        this.correctionRequest = {
          candidateId: candidate.candidateId,
          text: correctedText,
          key: crypto.randomUUID(),
        };
      }
      await firstValueFrom(
        this.api.correctInterpretation(
          interpretation.interpretationId,
          candidate.candidateId,
          correctedText,
          this.sessionId,
          this.correctionRequest.key,
        ),
      );
      this.correctionRequest = null;
      this.confirmedText = correctedText;
      this.statusMessage = 'Patient-confirmed correction sent. The corrected text is shown below.';
      await this.refreshMessages();
      this.stage = 'conversation';
    });
  }

  async rejectCandidates(): Promise<void> {
    const interpretation = this.interpretation;
    if (!interpretation || interpretation.status !== 'UNVERIFIED') {
      this.error = 'There are no unverified candidates to reject.';
      return;
    }
    await this.performInterpretationAction(async () => {
      await firstValueFrom(
        this.api.rejectInterpretation(interpretation.interpretationId, this.sessionId),
      );
      this.statusMessage = 'Suggestions rejected. No patient-confirmed message was created.';
      this.stage = 'conversation';
      await this.refreshMessages();
    });
  }

  trySigningAgain(): void {
    this.error = '';
    this.statusMessage =
      'Ready for another signing attempt. No new recognition has been submitted.';
    this.interpretation = null;
    this.selectedCandidateId = null;
    this.clearRecognitionSelection();
    this.releaseRecordedClip();
    this.stage = this.cameraAvailable ? 'capture' : 'setup';
  }

  useTypedCommunication(): void {
    this.discardRecording();
    this.releaseRecordedClip();
    this.stopCamera();
    this.stage = 'conversation';
    this.connectNotifications();
    void this.refreshMessages();
  }

  beginSigning(): void {
    if (!this.liveConsentRecorded) {
      this.stage = 'consent';
      return;
    }
    this.stage = this.cameraAvailable ? 'capture' : 'setup';
  }

  async sendTypedMessage(): Promise<void> {
    await this.sendPatientMessage(this.messageText, 'TEXT');
  }

  async sendVisualResponse(text: string): Promise<void> {
    await this.sendPatientMessage(text, 'VISUAL_RESPONSE');
  }

  async requestInterpreter(): Promise<void> {
    await this.sendPatientMessage(
      'I would like to request an interpreter.',
      'TEXT',
      'Interpreter request recorded. This prototype does not book an interpreter.',
    );
  }

  async withdrawConsent(): Promise<void> {
    this.busy = true;
    this.error = '';
    try {
      await firstValueFrom(this.api.recordLiveConsent(this.consultationId, this.sessionId, false));
      this.disconnectNotifications();
      this.discardRecording();
      this.stopCamera();
      this.releaseRecordedClip();
      this.interpretation = null;
      this.stage = 'consent';
      this.consentAccepted = false;
      this.liveConsentRecorded = false;
      this.statusMessage = 'Video-processing consent withdrawn. Camera stopped.';
    } catch (error: unknown) {
      this.handleRequestError(error);
    } finally {
      this.busy = false;
      this.changeDetector.markForCheck();
    }
  }

  private async sendPatientMessage(
    text: string,
    messageType: 'TEXT' | 'VISUAL_RESPONSE',
    successMessage = 'Your message was sent to the conversation.',
  ): Promise<void> {
    if (this.consultationClosed) {
      this.error = 'This consultation is read-only. You cannot send new messages.';
      this.changeDetector.markForCheck();
      return;
    }
    const value = text.trim();
    if (!value || !this.conversationId || !this.sessionId) {
      this.error = 'Enter a message before sending.';
      return;
    }
    this.busy = true;
    this.error = '';
    try {
      await firstValueFrom(
        this.api.sendMessage(this.conversationId, this.sessionId, value, messageType),
      );
      this.messageText = '';
      this.confirmedText = messageType === 'TEXT' ? value : '';
      this.statusMessage = successMessage;
      this.stage = 'conversation';
      await this.refreshMessages();
    } catch (error: unknown) {
      this.handleRequestError(error);
    } finally {
      this.busy = false;
      this.changeDetector.markForCheck();
    }
  }

  private async performInterpretationAction(action: () => Promise<void>): Promise<void> {
    this.busy = true;
    this.error = '';
    try {
      await action();
    } catch (error: unknown) {
      this.handleRequestError(error);
    } finally {
      this.busy = false;
      this.changeDetector.markForCheck();
    }
  }

  async refreshMessages(): Promise<void> {
    if (!this.conversationId || !this.sessionId) {
      return;
    }
    try {
      const result = await firstValueFrom(
        this.api.getMessages(this.conversationId, this.sessionId),
      );
      this.messages = [...result.messages].reverse();
      this.changeDetector.markForCheck();
    } catch (error: unknown) {
      this.handleRequestError(error);
      this.changeDetector.markForCheck();
    }
  }

  private connectNotifications(): void {
    this.disconnectNotifications();
    this.reconnectDelay = 1000;
    this.openNotificationSocket();
  }

  private openNotificationSocket(): void {
    if (!this.conversationId || !this.canReceiveNotifications()) {
      return;
    }
    try {
      this.socket = this.api.openNotifications(this.consultationId, this.sessionId);
      this.socket.onopen = () => {
        this.socketConnected = true;
        this.reconnectDelay = 1000;
        this.changeDetector.markForCheck();
        void this.refreshMessages();
      };
      this.socket.onmessage = async () => {
        this.changeDetector.markForCheck();
        await this.refreshConsultationStatus();
        await this.refreshMessages();
      };
      this.socket.onerror = () => {
        this.socketConnected = false;
        this.changeDetector.markForCheck();
      };
      this.socket.onclose = async () => {
        this.socketConnected = false;
        this.changeDetector.markForCheck();
        if (!this.canReceiveNotifications()) {
          return;
        }
        if (await this.refreshConsultationStatus()) {
          return;
        }
        if (!this.canReceiveNotifications()) {
          return;
        }
        const delay = this.reconnectDelay;
        this.reconnectDelay = Math.min(this.reconnectDelay * 2, 15000);
        this.reconnectTimer = setTimeout(() => this.openNotificationSocket(), delay);
      };
    } catch {
      this.socketConnected = false;
      this.statusMessage =
        'Live updates are unavailable. Messages can still be sent and are refreshed after each action.';
      this.changeDetector.markForCheck();
    }
  }

  private disconnectNotifications(): void {
    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.socket) {
      this.socket.onclose = null;
      this.socket.close();
      this.socket = null;
    }
    this.socketConnected = false;
  }

  private canReceiveNotifications(): boolean {
    return (
      this.stage !== 'join' &&
      this.sessionId.length > 0 &&
      !this.consultationClosed &&
      !this.destroyed
    );
  }

  private async refreshConsultationStatus(): Promise<boolean> {
    if (!this.consultationId) {
      return false;
    }
    try {
      const consultation = await firstValueFrom(this.api.getConsultation(this.consultationId));
      if (consultation.status === 'COMPLETED') {
        this.markConsultationReadOnly(
          'This consultation is complete. Its conversation is read-only.',
        );
        return true;
      }
      if (consultation.isExpired) {
        this.markConsultationReadOnly(
          'This consultation has expired. Its conversation is read-only.',
        );
        return true;
      }
      return false;
    } catch (error: unknown) {
      this.handleRequestError(error);
      this.changeDetector.markForCheck();
      return false;
    }
  }

  private handleRecognitionPollingError(error: unknown): void {
    if (isSessionRecoveryError(error)) {
      this.handleRequestError(error);
      return;
    }
    this.error = `Recognition status checking was interrupted. Choose Check recognition status later on this page to resume the same request without uploading again. Refreshing loses this temporary session, so rejoin before continuing. ${errorMessage(error)}`;
    this.statusMessage = '';
  }

  private handleRequestError(error: unknown): void {
    this.error = errorMessage(error);
    if (!isSessionRecoveryError(error)) {
      return;
    }

    this.disconnectNotifications();
    this.discardRecording();
    this.stopCamera();
    this.releaseRecordedClip();
    this.sessionId = '';
    this.conversationId = '';
    this.joinCode = '';
    this.liveConsentRecorded = false;
    this.consentAccepted = false;
    this.consultationClosed = false;
    this.clearRecognitionSelection();
    this.interpretation = null;
    this.selectedCandidateId = null;
    this.confirmationRequest = null;
    this.correctionRequest = null;
    this.messages = [];
    this.messageText = '';
    this.confirmedText = '';
    this.stage = 'join';
    this.statusMessage =
      'The temporary session was cleared. Rejoin with the consultation ID and join code. If the consultation has expired, ask the clinician to start a new one.';
    this.changeDetector.markForCheck();
  }

  private clearRecognitionSelection(): void {
    this.recognitionFile = null;
    this.recognitionFileRequestId = null;
    this.pendingRecognition = null;
    this.recognitionFileError = '';
    this.recognitionFileStatus = '';
    if (this.recognitionVideoInput) {
      this.recognitionVideoInput.nativeElement.value = '';
    }
  }

  private markConsultationReadOnly(message: string): void {
    this.consultationClosed = true;
    this.statusMessage = message;
    this.stage = 'conversation';
    this.disconnectNotifications();
    this.discardRecording();
    this.stopCamera();
    this.releaseRecordedClip();
    this.changeDetector.markForCheck();
  }

  private cameraPermissionMessage(error: unknown): string {
    if (error instanceof DOMException && error.name === 'NotAllowedError') {
      return 'Camera permission was denied. Allow camera access in browser settings and retry, or send a typed message.';
    }
    if (error instanceof DOMException && error.name === 'NotFoundError') {
      return 'No camera was found. Connect a camera or send a typed message.';
    }
    return 'Camera could not be opened. Check browser permission and device connection, then retry or send a typed message.';
  }

  private stopCamera(): void {
    this.cameraStream?.getTracks().forEach((track) => track.stop());
    this.cameraStream = null;
    this.cameraAvailable = false;
  }

  private discardRecording(): void {
    if (this.recordingTimer !== null) {
      clearInterval(this.recordingTimer);
      this.recordingTimer = null;
    }
    if (this.recorder) {
      this.recorder.onstop = null;
      this.recorder.ondataavailable = null;
      if (this.recorder.state === 'recording') {
        this.recorder.stop();
      }
      this.recorder = null;
    }
    this.recordingChunks = [];
    this.isRecording = false;
  }

  private releaseRecordedClip(): void {
    if (this.clipUrl) {
      URL.revokeObjectURL(this.clipUrl);
      this.clipUrl = null;
    }
    this.recordedClip = null;
  }

  ngOnDestroy(): void {
    this.destroyed = true;
    this.disconnectNotifications();
    this.discardRecording();
    this.stopCamera();
    this.releaseRecordedClip();
  }
}
