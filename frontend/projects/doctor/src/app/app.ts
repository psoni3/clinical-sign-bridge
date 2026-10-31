import { HttpErrorResponse } from '@angular/common/http';
import { DatePipe } from '@angular/common';
import { ChangeDetectorRef, Component, OnDestroy, OnInit, inject } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { firstValueFrom } from 'rxjs';

import {
  DoctorApiService,
  DoctorConsultationDto,
  DoctorConsultationInfoDto,
  MessageDto,
} from './doctor-api.service';

const CONSULTATION_REFRESH_MS = 10_000;

function sessionStorageKey(consultationId: string): string {
  return `clinical-sign-bridge.doctor-session.${consultationId}`;
}

function errorMessage(error: unknown): string {
  if (error instanceof HttpErrorResponse) {
    const payload: unknown = error.error;
    if (typeof payload === 'object' && payload !== null) {
      if ('detail' in payload && typeof payload.detail === 'object' && payload.detail !== null) {
        const detail = payload.detail;
        if ('code' in detail && typeof detail.code === 'string') {
          return `The service could not complete the request (${detail.code}).`;
        }
      }
      if ('message' in payload && typeof payload.message === 'string') {
        return payload.message;
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
  imports: [DatePipe, FormsModule],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App implements OnInit, OnDestroy {
  private readonly api = inject(DoctorApiService);
  private readonly changeDetector = inject(ChangeDetectorRef);
  private refreshTimer: ReturnType<typeof setInterval> | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectDelay = 1000;
  private socket: WebSocket | null = null;
  private destroyed = false;
  private restoreAttempted = false;

  consultations: DoctorConsultationDto[] = [];
  consultationLabel = '';
  expiresInMinutes = 120;
  consultationIdInput = '';
  joinCodeInput = '';
  createdConsultationId = '';
  createdJoinCode = '';
  consultation: DoctorConsultationInfoDto | null = null;
  doctorSessionId = '';
  messages: MessageDto[] = [];
  questionText = '';
  error = '';
  statusMessage = '';
  busy = false;
  socketConnected = false;

  get visibleMessages(): MessageDto[] {
    return this.messages.filter(
      (message) =>
        message.senderType === 'DOCTOR' || message.verificationStatus === 'PATIENT_CONFIRMED',
    );
  }

  get canStartConsultation(): boolean {
    return this.consultation?.status === 'JOINED';
  }

  get canSendQuestion(): boolean {
    return this.consultation?.status === 'ACTIVE' && Boolean(this.doctorSessionId);
  }

  get canCompleteConsultation(): boolean {
    return this.consultation?.status === 'ACTIVE';
  }

  get isCompleted(): boolean {
    return this.consultation?.status === 'COMPLETED';
  }

  ngOnInit(): void {
    void this.loadConsultations();
    this.refreshTimer = setInterval(() => void this.loadConsultations(), CONSULTATION_REFRESH_MS);
  }

  async loadConsultations(): Promise<void> {
    try {
      const result = await firstValueFrom(this.api.getConsultations());
      this.consultations = result.consultations;
      this.changeDetector.markForCheck();

      if (this.consultation) {
        const latest = this.consultations.find(
          (item) => item.consultationId === this.consultation?.consultationId,
        );
        if (latest) {
          this.consultation = { ...this.consultation, status: latest.status };
        }
        if (this.doctorSessionId) {
          await this.refreshCurrentConsultation();
        }
      } else if (!this.restoreAttempted) {
        this.restoreAttempted = true;
        await this.restoreLastConsultation();
      }
    } catch (error: unknown) {
      this.error = errorMessage(error);
      this.changeDetector.markForCheck();
    }
  }

  async createConsultation(): Promise<void> {
    this.error = '';
    this.statusMessage = '';
    const expiry = Number(this.expiresInMinutes);
    if (!Number.isInteger(expiry) || expiry < 1 || expiry > 1440) {
      this.error = 'Choose an expiry between 1 and 1,440 minutes.';
      return;
    }

    this.busy = true;
    this.createdConsultationId = '';
    this.createdJoinCode = '';
    try {
      const created = await firstValueFrom(
        this.api.createConsultation(this.consultationLabel.trim() || null, expiry),
      );
      this.createdConsultationId = created.consultationId;
      this.createdJoinCode = created.joinCode;
      this.consultationLabel = '';
      this.statusMessage =
        'Consultation created. Share the ID and one-time join code with the patient.';
      this.changeDetector.markForCheck();
      await this.connectToConsultation(created.consultationId, created.joinCode);
      await this.loadConsultations();
    } catch (error: unknown) {
      this.error = errorMessage(error);
      if (this.createdConsultationId && this.createdJoinCode) {
        this.error += ' The consultation and one-time code are shown below.';
      }
    } finally {
      this.busy = false;
      this.changeDetector.markForCheck();
    }
  }

  async connectUsingCode(): Promise<void> {
    this.error = '';
    const consultationId = this.consultationIdInput.trim();
    const joinCode = this.joinCodeInput.trim();
    if (!consultationId || !joinCode) {
      this.error = 'Enter the consultation ID and the join code.';
      return;
    }

    this.busy = true;
    try {
      await this.connectToConsultation(consultationId, joinCode);
      this.joinCodeInput = '';
      this.consultationIdInput = '';
      await this.loadConsultations();
    } catch (error: unknown) {
      this.error = errorMessage(error);
      this.changeDetector.markForCheck();
    } finally {
      this.busy = false;
      this.changeDetector.markForCheck();
    }
  }

  async openConsultation(item: DoctorConsultationDto): Promise<void> {
    this.error = '';
    const sessionId = sessionStorage.getItem(sessionStorageKey(item.consultationId));
    if (!sessionId) {
      this.consultationIdInput = item.consultationId;
      this.statusMessage = 'Enter the consultation join code to open this consultation.';
      return;
    }

    this.busy = true;
    try {
      await this.activateSession(item.consultationId, sessionId);
    } catch (error: unknown) {
      this.error = errorMessage(error);
      this.changeDetector.markForCheck();
    } finally {
      this.busy = false;
      this.changeDetector.markForCheck();
    }
  }

  async startConsultation(): Promise<void> {
    const consultation = this.consultation;
    if (!consultation || !this.canStartConsultation) {
      this.error = 'The patient must join before the consultation can be started.';
      return;
    }
    await this.runAction(async () => {
      await firstValueFrom(this.api.startConsultation(consultation.consultationId));
      await this.refreshCurrentConsultation();
      await this.loadConsultations();
      if (!this.error) {
        this.statusMessage = 'Consultation started.';
      }
    });
  }

  async completeConsultation(): Promise<void> {
    const consultation = this.consultation;
    if (!consultation || !this.canCompleteConsultation) {
      this.error = 'There is no active consultation to complete.';
      return;
    }
    await this.runAction(async () => {
      await firstValueFrom(this.api.completeConsultation(consultation.consultationId));
      await this.refreshCurrentConsultation();
      await this.loadConsultations();
      if (!this.error) {
        this.statusMessage =
          'Consultation completed. Its conversation remains available read-only.';
      }
    });
  }

  async sendQuestion(): Promise<void> {
    const text = this.questionText.trim();
    const consultation = this.consultation;
    if (!text || !consultation || !this.canSendQuestion) {
      this.error = 'Start the consultation and enter a question before sending.';
      return;
    }
    await this.runAction(async () => {
      await firstValueFrom(
        this.api.sendDoctorMessage(consultation.conversationId, this.doctorSessionId, text),
      );
      this.questionText = '';
      this.changeDetector.markForCheck();
      await this.refreshMessages();
      if (!this.error) {
        this.statusMessage = 'Your message was sent to the patient.';
      }
    });
  }

  async acknowledgeMessage(message: MessageDto): Promise<void> {
    if (
      !this.consultation ||
      this.isCompleted ||
      message.senderType !== 'PATIENT' ||
      message.verificationStatus !== 'PATIENT_CONFIRMED' ||
      message.acknowledgedAt
    ) {
      return;
    }
    await this.runAction(async () => {
      await firstValueFrom(this.api.acknowledgeMessage(message.messageId, this.doctorSessionId));
      await this.refreshMessages();
      if (!this.error) {
        this.statusMessage = 'Message marked clinician-reviewed.';
        this.changeDetector.markForCheck();
      }
    });
  }

  async copyJoinDetails(): Promise<void> {
    if (!this.createdConsultationId || !this.createdJoinCode) {
      return;
    }
    try {
      await navigator.clipboard.writeText(
        `Consultation ID: ${this.createdConsultationId}\nJoin code: ${this.createdJoinCode}`,
      );
      this.statusMessage = 'Consultation ID and join code copied.';
      this.changeDetector.markForCheck();
    } catch {
      this.error = 'Could not copy details. Select and copy the consultation ID and code manually.';
      this.changeDetector.markForCheck();
    }
  }

  async refreshCurrentConsultation(): Promise<void> {
    const consultation = this.consultation;
    if (!consultation) {
      return;
    }
    try {
      const latest = await firstValueFrom(this.api.getConsultation(consultation.consultationId));
      this.consultation = latest;
      this.changeDetector.markForCheck();
      if (latest.status === 'COMPLETED') {
        this.disconnectNotifications();
        await this.refreshMessages();
      } else if (latest.isExpired) {
        this.disconnectNotifications();
      } else if (latest.status === 'CREATED') {
        this.disconnectNotifications();
      } else {
        if (!this.socket) {
          this.connectNotifications();
        }
        await this.refreshMessages();
      }
    } catch (error: unknown) {
      this.error = errorMessage(error);
      this.changeDetector.markForCheck();
    }
  }

  async refreshMessages(): Promise<void> {
    const consultation = this.consultation;
    if (!consultation || !this.doctorSessionId) {
      return;
    }
    try {
      const result = await firstValueFrom(
        this.api.getMessages(consultation.conversationId, this.doctorSessionId),
      );
      this.messages = [...result.messages].reverse();
      this.changeDetector.markForCheck();
    } catch (error: unknown) {
      this.error = errorMessage(error);
      this.changeDetector.markForCheck();
    }
  }

  provenanceLabel(message: MessageDto): string {
    if (message.verificationStatus === 'PATIENT_CONFIRMED') {
      return message.provenance === 'AI_CANDIDATE_CONFIRMED'
        ? 'Patient-confirmed interpretation'
        : message.provenance === 'PATIENT_CORRECTED'
          ? 'Patient-confirmed correction'
          : message.provenance === 'PATIENT_VISUAL_RESPONSE'
            ? 'Patient-selected visual response'
            : 'Patient-typed message';
    }
    return message.provenance.replaceAll('_', ' ').toLowerCase();
  }

  ngOnDestroy(): void {
    this.destroyed = true;
    if (this.refreshTimer !== null) {
      clearInterval(this.refreshTimer);
    }
    this.disconnectNotifications();
  }

  private async restoreLastConsultation(): Promise<void> {
    const consultationId = sessionStorage.getItem('clinical-sign-bridge.doctor-selected');
    if (!consultationId) {
      return;
    }
    const sessionId = sessionStorage.getItem(sessionStorageKey(consultationId));
    if (!sessionId) {
      return;
    }
    try {
      await this.activateSession(consultationId, sessionId);
    } catch (error: unknown) {
      this.error = errorMessage(error);
    }
  }

  private async connectToConsultation(consultationId: string, joinCode: string): Promise<void> {
    const joined = await firstValueFrom(this.api.joinConsultation(consultationId, joinCode));
    sessionStorage.setItem(sessionStorageKey(consultationId), joined.sessionId);
    await this.activateSession(consultationId, joined.sessionId);
  }

  private async activateSession(consultationId: string, sessionId: string): Promise<void> {
    const details = await firstValueFrom(this.api.getConsultation(consultationId));
    this.disconnectNotifications();
    this.consultation = details;
    this.doctorSessionId = sessionId;
    this.messages = [];
    sessionStorage.setItem('clinical-sign-bridge.doctor-selected', consultationId);
    this.error = '';
    this.statusMessage = `Connected to ${details.displayLabel || 'consultation'}.`;
    this.changeDetector.markForCheck();
    await this.refreshCurrentConsultation();
  }

  private connectNotifications(): void {
    if (
      !this.consultation ||
      !this.doctorSessionId ||
      this.destroyed ||
      !['JOINED', 'ACTIVE'].includes(this.consultation.status)
    ) {
      return;
    }
    try {
      this.socket = this.api.openNotifications(
        this.consultation.consultationId,
        this.doctorSessionId,
      );
      this.socket.onopen = () => {
        this.socketConnected = true;
        this.reconnectDelay = 1000;
        this.changeDetector.markForCheck();
        void this.refreshCurrentConsultation();
      };
      this.socket.onmessage = () => {
        this.changeDetector.markForCheck();
        void this.refreshCurrentConsultation();
        void this.loadConsultations();
      };
      this.socket.onerror = () => {
        this.socketConnected = false;
        this.changeDetector.markForCheck();
      };
      this.socket.onclose = () => {
        this.socketConnected = false;
        this.changeDetector.markForCheck();
        if (!this.destroyed && this.consultation && this.doctorSessionId) {
          const delay = this.reconnectDelay;
          this.reconnectDelay = Math.min(this.reconnectDelay * 2, 15_000);
          this.reconnectTimer = setTimeout(() => this.connectNotifications(), delay);
        }
      };
    } catch {
      this.socketConnected = false;
      this.statusMessage =
        'Live updates are unavailable. The dashboard will refresh from the service periodically.';
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

  private async runAction(action: () => Promise<void>): Promise<void> {
    this.error = '';
    this.busy = true;
    try {
      await action();
    } catch (error: unknown) {
      this.error = errorMessage(error);
    } finally {
      this.busy = false;
      this.changeDetector.markForCheck();
    }
  }
}
