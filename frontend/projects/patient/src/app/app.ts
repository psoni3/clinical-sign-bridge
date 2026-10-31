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

function isPendingRecognition(status: RecognitionDto['status']): boolean {
  return status === 'QUEUED' || status === 'PROCESSING';
}

function errorMessage(error: unknown): string {
  if (error instanceof HttpErrorResponse) {
    const payload: unknown = error.error;
    if (typeof payload === 'object' && payload !== null && 'message' in payload) {
      const message = payload.message;
      if (typeof message === 'string') {
        return message;
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
  private recordingRequestId: string | null = null;
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
      this.error = errorMessage(error);
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
      this.statusMessage = 'Live-translation processing consent recorded for this consultation.';
      this.connectNotifications();
      await this.refreshMessages();
    } catch (error: unknown) {
      this.error = errorMessage(error);
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
      this.cameraError = 'Accept live-translation processing consent before enabling the camera.';
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
        'This segment is too large to upload. Record a shorter segment or use typed text.';
      this.changeDetector.markForCheck();
      return;
    }
    this.recordedClip = clip;
    this.recordingRequestId = crypto.randomUUID();
    this.clipUrl = URL.createObjectURL(clip);
    this.statusMessage = `Segment ready (${this.recordingSeconds} seconds). Review before uploading.`;
    this.changeDetector.markForCheck();
  }

  async uploadSegment(): Promise<void> {
    if (!this.recordedClip) {
      this.error = 'Record a segment before uploading.';
      return;
    }
    this.busy = true;
    this.error = '';
    this.statusMessage = 'Uploading the segment for processing...';
    try {
      let recognition = await firstValueFrom(
        this.api.uploadSegment(
          this.consultationId,
          this.sessionId,
          this.recordingRequestId ?? crypto.randomUUID(),
          this.recordedClip,
        ),
      );
      for (let attempt = 0; isPendingRecognition(recognition.status); attempt += 1) {
        if (attempt >= RECOGNITION_POLL_LIMIT) {
          this.error =
            'Processing is taking longer than expected. Retry by checking the consultation later.';
          return;
        }
        await new Promise((resolve) => setTimeout(resolve, 1000));
        recognition = await firstValueFrom(
          this.api.getRecognition(recognition.recognitionId, this.sessionId),
        );
        this.statusMessage = `Recognition status: ${recognition.status.toLowerCase()}.`;
      }
      if (!recognition.interpretationId) {
        this.error = 'No interpretation was returned. Use typed text or request an interpreter.';
        return;
      }
      this.interpretation = await firstValueFrom(
        this.api.getInterpretation(recognition.interpretationId, this.sessionId),
      );
      this.selectedCandidateId = null;
      this.correctionText = '';
      this.stage = 'review';
      this.statusMessage =
        recognition.status === 'UNSUPPORTED'
          ? 'The mock provider cannot recognize this segment. No text was guessed or generated.'
          : 'Review the unverified interpretation before choosing an action.';
      this.releaseRecordedClip();
    } catch (error: unknown) {
      this.error = errorMessage(error);
      this.statusMessage = '';
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
    this.statusMessage = '';
    this.interpretation = null;
    this.selectedCandidateId = null;
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
      this.statusMessage = 'Live-translation processing consent withdrawn. Camera stopped.';
    } catch (error: unknown) {
      this.error = errorMessage(error);
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
      this.error = errorMessage(error);
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
      this.error = errorMessage(error);
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
      this.error = errorMessage(error);
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
      this.error = errorMessage(error);
      this.changeDetector.markForCheck();
      return false;
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
    this.recordingRequestId = null;
  }

  ngOnDestroy(): void {
    this.destroyed = true;
    this.disconnectNotifications();
    this.discardRecording();
    this.stopCamera();
    this.releaseRecordedClip();
  }
}
