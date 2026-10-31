import { HttpClient, HttpParams } from '@angular/common/http';
import { inject, Injectable, InjectionToken } from '@angular/core';
import { Observable } from 'rxjs';

export const DOCTOR_API_BASE_URL = new InjectionToken<string>('DOCTOR_API_BASE_URL', {
  providedIn: 'root',
  factory: () => 'http://localhost:8000/api/v1',
});

export interface CreatedConsultationDto {
  consultationId: string;
  joinCode: string;
  status: string;
  expiresAt: string;
}

export interface DoctorSessionDto {
  consultationId: string;
  sessionId: string;
  sessionToken: string;
  status: string;
}

export interface DoctorConsultationDto {
  consultationId: string;
  displayLabel: string | null;
  status: string;
  createdAt: string;
  lastActivityAt: string;
  isActive: boolean;
}

export interface DoctorConsultationsDto {
  consultations: DoctorConsultationDto[];
  total: number;
}

export interface DoctorConsultationInfoDto {
  consultationId: string;
  conversationId: string;
  displayLabel: string | null;
  status: string;
  expiresAt: string;
  createdAt: string;
  startedAt: string | null;
  completedAt: string | null;
  isExpired: boolean;
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
export class DoctorApiService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = inject(DOCTOR_API_BASE_URL).replace(/\/$/, '');

  createConsultation(
    displayLabel: string | null,
    expiresInMinutes: number,
  ): Observable<CreatedConsultationDto> {
    return this.http.post<CreatedConsultationDto>(`${this.baseUrl}/consultations`, {
      displayLabel,
      expiresInMinutes,
    });
  }

  joinConsultation(consultationId: string, joinCode: string): Observable<DoctorSessionDto> {
    return this.http.post<DoctorSessionDto>(
      `${this.baseUrl}/consultations/${encodeURIComponent(consultationId)}/join`,
      { joinCode, deviceLabel: 'Clinician dashboard', sessionType: 'DOCTOR_VIEW' },
    );
  }

  getConsultations(): Observable<DoctorConsultationsDto> {
    const params = new HttpParams()
      .set('offset', 0)
      .set('limit', 100)
      .set('includeCompleted', true);
    return this.http.get<DoctorConsultationsDto>(`${this.baseUrl}/doctors/consultations`, {
      params,
    });
  }

  getConsultation(consultationId: string): Observable<DoctorConsultationInfoDto> {
    return this.http.get<DoctorConsultationInfoDto>(
      `${this.baseUrl}/consultations/${encodeURIComponent(consultationId)}`,
    );
  }

  startConsultation(consultationId: string): Observable<unknown> {
    return this.http.post(
      `${this.baseUrl}/consultations/${encodeURIComponent(consultationId)}/start`,
      {},
    );
  }

  completeConsultation(consultationId: string): Observable<unknown> {
    return this.http.post(
      `${this.baseUrl}/consultations/${encodeURIComponent(consultationId)}/complete`,
      {},
    );
  }

  getMessages(conversationId: string, sessionId: string): Observable<MessagesDto> {
    const params = new HttpParams().set('sessionId', sessionId).set('limit', 100);
    return this.http.get<MessagesDto>(
      `${this.baseUrl}/conversations/${encodeURIComponent(conversationId)}/messages`,
      { params },
    );
  }

  sendDoctorMessage(conversationId: string, sessionId: string, text: string): Observable<unknown> {
    return this.http.post(
      `${this.baseUrl}/conversations/${encodeURIComponent(conversationId)}/messages`,
      { senderType: 'DOCTOR', sessionId, messageType: 'TEXT', text },
    );
  }

  acknowledgeMessage(messageId: string, sessionId: string): Observable<unknown> {
    return this.http.post(`${this.baseUrl}/messages/${encodeURIComponent(messageId)}/acknowledge`, {
      sessionId,
    });
  }

  openNotifications(consultationId: string, sessionId: string): WebSocket {
    const url = new URL(this.baseUrl);
    url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
    url.pathname = `${url.pathname.replace(/\/$/, '')}/ws/consultations/${encodeURIComponent(consultationId)}`;
    url.searchParams.set('sessionId', sessionId);
    return new WebSocket(url);
  }
}
