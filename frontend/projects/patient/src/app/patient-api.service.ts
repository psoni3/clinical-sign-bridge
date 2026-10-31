import { HttpClient, HttpParams } from '@angular/common/http';
import { inject, Injectable, InjectionToken } from '@angular/core';
import { Observable } from 'rxjs';

export const PATIENT_API_BASE_URL = new InjectionToken<string>('PATIENT_API_BASE_URL', {
  providedIn: 'root',
  factory: () => 'http://localhost:8000/api/v1',
});

export interface ConsultationDto {
  consultationId: string;
  conversationId: string;
  status: string;
  expiresAt: string;
  isExpired: boolean;
}

export interface JoinConsultationDto {
  consultationId: string;
  sessionId: string;
  sessionToken: string;
  status: string;
}

export interface RecognitionDto {
  recognitionId: string;
  status: 'QUEUED' | 'PROCESSING' | 'COMPLETED' | 'FAILED' | 'UNSUPPORTED';
  interpretationId: string | null;
}

export interface CandidateDto {
  candidateId: string;
  rank: number;
  text: string;
  confidence: number | null;
}

export interface InterpretationDto {
  interpretationId: string;
  status: 'UNVERIFIED' | 'CONFIRMED' | 'REJECTED' | 'NEEDS_CLARIFICATION';
  modelVersion: string;
  quality: { status: string; messages: string[] };
  candidates: CandidateDto[];
  availableActions: string[];
}

export interface MessageDto {
  messageId: string;
  senderType: 'DOCTOR' | 'PATIENT';
  messageType: 'TEXT' | 'VISUAL_RESPONSE' | 'AI_TRANSLATION';
  text: string;
  provenance: string;
  verificationStatus: string;
  createdAt: string;
  confirmedAt: string | null;
  acknowledgedAt: string | null;
}

export interface MessagesDto {
  messages: MessageDto[];
  total: number;
  offset: number;
  limit: number;
}

@Injectable({ providedIn: 'root' })
export class PatientApiService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = inject(PATIENT_API_BASE_URL).replace(/\/$/, '');

  joinConsultation(consultationId: string, joinCode: string): Observable<JoinConsultationDto> {
    return this.http.post<JoinConsultationDto>(
      `${this.baseUrl}/consultations/${encodeURIComponent(consultationId)}/join`,
      { joinCode, deviceLabel: 'Patient device', sessionType: 'PATIENT_DEVICE' },
    );
  }

  getConsultation(consultationId: string): Observable<ConsultationDto> {
    return this.http.get<ConsultationDto>(
      `${this.baseUrl}/consultations/${encodeURIComponent(consultationId)}`,
    );
  }

  recordLiveConsent(
    consultationId: string,
    sessionId: string,
    accepted: boolean,
  ): Observable<unknown> {
    return this.http.post(
      `${this.baseUrl}/consultations/${encodeURIComponent(consultationId)}/consents`,
      {
        sessionId,
        consentType: 'PROCESSING_FOR_LIVE_TRANSLATION',
        policyVersion: 'm2-v1',
        accepted,
      },
    );
  }

  uploadSegment(
    consultationId: string,
    sessionId: string,
    clientRequestId: string,
    segment: Blob,
  ): Observable<RecognitionDto> {
    const form = new FormData();
    form.append('video', segment, 'signing-segment.webm');
    form.append('clientRequestId', clientRequestId);
    form.append('sessionId', sessionId);
    return this.http.post<RecognitionDto>(
      `${this.baseUrl}/consultations/${encodeURIComponent(consultationId)}/recognitions`,
      form,
    );
  }

  getRecognition(recognitionId: string, sessionId: string): Observable<RecognitionDto> {
    const params = new HttpParams().set('sessionId', sessionId);
    return this.http.get<RecognitionDto>(
      `${this.baseUrl}/recognitions/${encodeURIComponent(recognitionId)}`,
      { params },
    );
  }

  getInterpretation(interpretationId: string, sessionId: string): Observable<InterpretationDto> {
    const params = new HttpParams().set('sessionId', sessionId);
    return this.http.get<InterpretationDto>(
      `${this.baseUrl}/interpretations/${encodeURIComponent(interpretationId)}`,
      { params },
    );
  }

  confirmInterpretation(
    interpretationId: string,
    candidateId: string,
    sessionId: string,
    idempotencyKey: string,
  ): Observable<unknown> {
    return this.http.post(
      `${this.baseUrl}/interpretations/${encodeURIComponent(interpretationId)}/confirm`,
      { candidateId, sessionId, idempotencyKey },
    );
  }

  correctInterpretation(
    interpretationId: string,
    candidateId: string,
    correctedText: string,
    sessionId: string,
    idempotencyKey: string,
  ): Observable<unknown> {
    return this.http.post(
      `${this.baseUrl}/interpretations/${encodeURIComponent(interpretationId)}/correct`,
      { candidateId, correctedText, sessionId, idempotencyKey },
    );
  }

  rejectInterpretation(interpretationId: string, sessionId: string): Observable<unknown> {
    return this.http.post(
      `${this.baseUrl}/interpretations/${encodeURIComponent(interpretationId)}/reject`,
      { sessionId, reason: 'NONE_OF_THESE' },
    );
  }

  getMessages(conversationId: string, sessionId: string): Observable<MessagesDto> {
    const params = new HttpParams().set('sessionId', sessionId).set('limit', 100);
    return this.http.get<MessagesDto>(
      `${this.baseUrl}/conversations/${encodeURIComponent(conversationId)}/messages`,
      { params },
    );
  }

  sendMessage(
    conversationId: string,
    sessionId: string,
    text: string,
    messageType: 'TEXT' | 'VISUAL_RESPONSE',
  ): Observable<unknown> {
    return this.http.post(
      `${this.baseUrl}/conversations/${encodeURIComponent(conversationId)}/messages`,
      { senderType: 'PATIENT', sessionId, messageType, text },
    );
  }

  openNotifications(consultationId: string, sessionId: string): WebSocket {
    const url = new URL(this.baseUrl);
    url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
    url.pathname = `${url.pathname.replace(/\/$/, '')}/ws/consultations/${encodeURIComponent(consultationId)}`;
    url.searchParams.set('sessionId', sessionId);
    return new WebSocket(url);
  }
}
