import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';

import { DOCTOR_API_BASE_URL, DoctorApiService } from './doctor-api.service';

describe('DoctorApiService', () => {
  let api: DoctorApiService;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
        { provide: DOCTOR_API_BASE_URL, useValue: 'http://localhost:8000/api/v1' },
      ],
    });
    api = TestBed.inject(DoctorApiService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  it('creates a consultation using camelCase fields', () => {
    api.createConsultation('Room test', 120).subscribe();
    const request = http.expectOne('http://localhost:8000/api/v1/consultations');

    expect(request.request.method).toBe('POST');
    expect(request.request.body).toEqual({ displayLabel: 'Room test', expiresInMinutes: 120 });
    request.flush({
      consultationId: 'consultation-1',
      joinCode: 'ABCDEF012345',
      status: 'CREATED',
      expiresAt: '2026-10-31T12:00:00Z',
    });
  });

  it('joins a doctor view using the supplied join code', () => {
    api.joinConsultation('consultation-1', 'ABCDEF012345').subscribe();
    const request = http.expectOne(
      'http://localhost:8000/api/v1/consultations/consultation-1/join',
    );

    expect(request.request.method).toBe('POST');
    expect(request.request.body).toEqual({
      joinCode: 'ABCDEF012345',
      deviceLabel: 'Clinician dashboard',
      sessionType: 'DOCTOR_VIEW',
    });
    request.flush({
      consultationId: 'consultation-1',
      sessionId: 'session-1',
      sessionToken: 'unused-prototype-token',
      status: 'JOINED',
    });
  });

  it('includes completed consultations and uses the doctor session for history', () => {
    api.getConsultations().subscribe();
    const listRequest = http.expectOne(
      (request) =>
        request.url === 'http://localhost:8000/api/v1/doctors/consultations' &&
        request.params.get('includeCompleted') === 'true' &&
        request.params.get('limit') === '100',
    );
    listRequest.flush({ consultations: [], total: 0 });

    api.getMessages('conversation-1', 'session-1').subscribe();
    const historyRequest = http.expectOne(
      (request) =>
        request.url === 'http://localhost:8000/api/v1/conversations/conversation-1/messages' &&
        request.params.get('sessionId') === 'session-1' &&
        request.params.get('limit') === '100',
    );
    historyRequest.flush({ messages: [], total: 0, offset: 0, limit: 100 });
  });

  it('sends doctor questions and acknowledgements with the doctor session', () => {
    api.sendDoctorMessage('conversation-1', 'session-1', 'Where is the pain?').subscribe();
    const message = http.expectOne(
      'http://localhost:8000/api/v1/conversations/conversation-1/messages',
    );
    expect(message.request.body).toEqual({
      senderType: 'DOCTOR',
      sessionId: 'session-1',
      messageType: 'TEXT',
      text: 'Where is the pain?',
    });
    message.flush({ messageId: 'message-1', conversationId: 'conversation-1' });

    api.acknowledgeMessage('message-2', 'session-1').subscribe();
    const acknowledgement = http.expectOne(
      'http://localhost:8000/api/v1/messages/message-2/acknowledge',
    );
    expect(acknowledgement.request.body).toEqual({ sessionId: 'session-1' });
    acknowledgement.flush({ messageId: 'message-2', acknowledgedAt: '2026-10-31T12:00:00Z' });
  });
});
