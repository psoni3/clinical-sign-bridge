import { TestBed } from '@angular/core/testing';
import { of } from 'rxjs';

import { App } from './app';
import { DoctorApiService, MessageDto } from './doctor-api.service';

const consultation = {
  consultationId: 'consultation-1',
  conversationId: 'conversation-1',
  displayLabel: 'Synthetic room',
  status: 'CREATED',
  expiresAt: '2026-10-31T13:00:00Z',
  createdAt: '2026-10-31T11:00:00Z',
  startedAt: null,
  completedAt: null,
  isExpired: false,
};

function connectedSocket(): WebSocket {
  return {
    close: vi.fn(),
    onclose: null,
    onerror: null,
    onmessage: null,
    onopen: null,
  } as unknown as WebSocket;
}

describe('Doctor application', () => {
  it('renders consultation creation, connection, and prototype warnings', async () => {
    const api = { getConsultations: vi.fn().mockReturnValue(of({ consultations: [], total: 0 })) };
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [{ provide: DoctorApiService, useValue: api }],
    }).compileComponents();
    const fixture = TestBed.createComponent(App);
    fixture.detectChanges();

    expect(fixture.nativeElement.textContent).toContain('Create a consultation');
    expect(fixture.nativeElement.textContent).toContain('Open an existing consultation');
    expect(fixture.nativeElement.textContent).toContain('Use synthetic information only');
    expect(fixture.nativeElement.querySelector('#consultation-expiry')).not.toBeNull();
  });

  it('creates a consultation, opens a doctor session, and retains the one-time join code', async () => {
    const api = {
      createConsultation: vi.fn().mockReturnValue(
        of({
          consultationId: 'consultation-1',
          joinCode: 'ABCDEF012345',
          status: 'CREATED',
          expiresAt: '2026-10-31T13:00:00Z',
        }),
      ),
      joinConsultation: vi.fn().mockReturnValue(
        of({
          consultationId: 'consultation-1',
          sessionId: 'doctor-session-1',
          sessionToken: 'unused',
          status: 'CREATED',
        }),
      ),
      getConsultation: vi.fn().mockReturnValue(of(consultation)),
      getConsultations: vi.fn().mockReturnValue(
        of({
          consultations: [
            {
              consultationId: 'consultation-1',
              displayLabel: 'Synthetic room',
              status: 'CREATED',
              createdAt: '2026-10-31T11:00:00Z',
              lastActivityAt: '2026-10-31T11:00:00Z',
              isActive: true,
            },
          ],
          total: 1,
        }),
      ),
      openNotifications: vi.fn().mockReturnValue(connectedSocket()),
    };
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [{ provide: DoctorApiService, useValue: api }],
    }).compileComponents();
    const fixture = TestBed.createComponent(App);
    const component = fixture.componentInstance;

    await component.createConsultation();

    expect(api.createConsultation).toHaveBeenCalledWith(null, 120);
    expect(api.joinConsultation).toHaveBeenCalledWith('consultation-1', 'ABCDEF012345');
    expect(component.consultation?.status).toBe('CREATED');
    expect(component.createdJoinCode).toBe('ABCDEF012345');
    expect(component.canStartConsultation).toBe(false);
    expect(api.openNotifications).not.toHaveBeenCalled();
  });

  it('hides unverified patient suggestions and labels confirmed provenance', async () => {
    const api = { getConsultations: vi.fn().mockReturnValue(of({ consultations: [], total: 0 })) };
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [{ provide: DoctorApiService, useValue: api }],
    }).compileComponents();
    const fixture = TestBed.createComponent(App);
    const component = fixture.componentInstance;
    const message = (verificationStatus: string, provenance: string): MessageDto => ({
      messageId: provenance,
      senderType: 'PATIENT',
      messageType: 'TEXT',
      text: provenance,
      provenance,
      verificationStatus,
      createdAt: '2026-10-31T11:00:00Z',
      confirmedAt: null,
      acknowledgedAt: null,
    });
    component.messages = [
      message('UNVERIFIED', 'AI_TRANSLATION'),
      message('PATIENT_CONFIRMED', 'PATIENT_CORRECTED'),
    ];

    expect(component.visibleMessages).toHaveLength(1);
    expect(component.provenanceLabel(component.visibleMessages[0])).toBe(
      'Patient-confirmed correction',
    );
  });

  it('enables the doctor question composer after consultation start', async () => {
    const api = { getConsultations: vi.fn().mockReturnValue(of({ consultations: [], total: 0 })) };
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [{ provide: DoctorApiService, useValue: api }],
    }).compileComponents();
    const fixture = TestBed.createComponent(App);
    const component = fixture.componentInstance;
    component.consultation = { ...consultation, status: 'ACTIVE' };
    component.doctorSessionId = 'doctor-session-1';

    expect(component.canSendQuestion).toBe(true);
    expect(component.canCompleteConsultation).toBe(true);
  });
});
