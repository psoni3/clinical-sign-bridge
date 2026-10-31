import { TestBed } from '@angular/core/testing';
import { of, throwError } from 'rxjs';

import { App } from './app';
import { PatientApiService } from './patient-api.service';

describe('Patient application', () => {
  it('shows the no-auth prototype notice and consultation join form', async () => {
    const api = {
      joinConsultation: vi.fn(),
      getConsultation: vi.fn(),
    };
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [{ provide: PatientApiService, useValue: api }],
    }).compileComponents();
    const fixture = TestBed.createComponent(App);
    fixture.detectChanges();

    expect(fixture.nativeElement.textContent).toContain(
      'This application does not recognize Indian Sign Language.',
    );
    expect(fixture.nativeElement.textContent).toContain('Join a consultation');
    expect(fixture.nativeElement.textContent).toContain('No account or password is needed.');
    expect(fixture.nativeElement.querySelector('#consultation-id')).not.toBeNull();
    expect(fixture.nativeElement.querySelector('#join-code')).not.toBeNull();
  });

  it('joins with the supplied consultation and moves to consent', async () => {
    const api = {
      joinConsultation: vi.fn().mockReturnValue(
        of({
          consultationId: 'consultation-1',
          sessionId: 'session-1',
          sessionToken: 'temporary-token',
          status: 'JOINED',
        }),
      ),
      getConsultation: vi.fn().mockReturnValue(
        of({
          consultationId: 'consultation-1',
          conversationId: 'conversation-1',
          status: 'JOINED',
          expiresAt: '2026-10-31T12:00:00Z',
        }),
      ),
    };
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [{ provide: PatientApiService, useValue: api }],
    }).compileComponents();
    const fixture = TestBed.createComponent(App);
    const component = fixture.componentInstance;
    component.consultationId = 'consultation-1';
    component.joinCode = 'ABCDEF012345';

    await component.join();

    expect(api.joinConsultation).toHaveBeenCalledWith('consultation-1', 'ABCDEF012345');
    expect(component.stage).toBe('consent');
    expect(component.sessionId).toBe('session-1');
    expect(component.conversationId).toBe('conversation-1');
  });

  it('does not request processing consent without an explicit checkbox action', async () => {
    const api = {
      recordLiveConsent: vi.fn(),
    };
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [{ provide: PatientApiService, useValue: api }],
    }).compileComponents();
    const fixture = TestBed.createComponent(App);

    await fixture.componentInstance.acceptLiveConsent();

    expect(api.recordLiveConsent).not.toHaveBeenCalled();
    expect(fixture.componentInstance.error).toContain('Select the consent checkbox');
  });

  it('reuses the same confirmation idempotency key after a retryable failure', async () => {
    const api = {
      confirmInterpretation: vi
        .fn()
        .mockReturnValueOnce(throwError(() => new Error('Temporary service error')))
        .mockReturnValueOnce(of({})),
      getMessages: vi.fn().mockReturnValue(of({ messages: [], total: 0, offset: 0, limit: 100 })),
    };
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [{ provide: PatientApiService, useValue: api }],
    }).compileComponents();
    const fixture = TestBed.createComponent(App);
    const component = fixture.componentInstance;
    component.sessionId = 'session-1';
    component.conversationId = 'conversation-1';
    component.interpretation = {
      interpretationId: 'interpretation-1',
      status: 'UNVERIFIED',
      modelVersion: 'mock-v1',
      quality: { status: 'ACCEPTABLE', messages: [] },
      candidates: [
        {
          candidateId: 'candidate-1',
          rank: 1,
          text: 'Example phrase',
          confidence: 0.5,
        },
      ],
      availableActions: ['CONFIRM'],
    };
    component.selectedCandidateId = 'candidate-1';

    await component.confirmSelectedCandidate();
    await component.confirmSelectedCandidate();

    expect(api.confirmInterpretation).toHaveBeenCalledTimes(2);
    expect(api.confirmInterpretation.mock.calls[0][3]).toBe(
      api.confirmInterpretation.mock.calls[1][3],
    );
    expect(component.confirmedText).toBe('Example phrase');
  });

  it('allows typed communication without recording video-processing consent', async () => {
    const api = {
      sendMessage: vi.fn().mockReturnValue(of({})),
      getMessages: vi.fn().mockReturnValue(of({ messages: [], total: 0, offset: 0, limit: 100 })),
      openNotifications: vi.fn().mockReturnValue({ close: vi.fn() }),
    };
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [{ provide: PatientApiService, useValue: api }],
    }).compileComponents();
    const fixture = TestBed.createComponent(App);
    const component = fixture.componentInstance;
    component.sessionId = 'session-1';
    component.conversationId = 'conversation-1';
    component.consentAccepted = false;
    component.liveConsentRecorded = false;
    component.messageText = 'Please repeat the question.';

    await component.sendTypedMessage();

    expect(api.sendMessage).toHaveBeenCalledWith(
      'conversation-1',
      'session-1',
      'Please repeat the question.',
      'TEXT',
    );
    expect(component.stage).toBe('conversation');
  });

  it('stops reconnecting and makes the conversation read-only when the consultation is complete', async () => {
    const socket: {
      close: ReturnType<typeof vi.fn>;
      onopen: (() => void) | null;
      onmessage: ((event: MessageEvent) => void | Promise<void>) | null;
      onerror: ((event: Event) => void) | null;
      onclose: (() => void | Promise<void>) | null;
    } = {
      close: vi.fn(),
      onopen: null,
      onmessage: null,
      onerror: null,
      onclose: null,
    };
    const api = {
      getConsultation: vi.fn().mockReturnValue(
        of({
          consultationId: 'consultation-1',
          conversationId: 'conversation-1',
          status: 'COMPLETED',
          expiresAt: '2026-10-31T12:00:00Z',
          isExpired: false,
        }),
      ),
      getMessages: vi.fn().mockReturnValue(of({ messages: [], total: 0, offset: 0, limit: 100 })),
      openNotifications: vi.fn().mockReturnValue(socket),
      sendMessage: vi.fn().mockReturnValue(of({})),
    };
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [{ provide: PatientApiService, useValue: api }],
    }).compileComponents();
    const fixture = TestBed.createComponent(App);
    const component = fixture.componentInstance;
    component.consultationId = 'consultation-1';
    component.sessionId = 'session-1';
    component.conversationId = 'conversation-1';
    component.stage = 'conversation';
    component.useTypedCommunication();

    const setTimeoutSpy = vi.spyOn(globalThis, 'setTimeout');
    try {
      await socket.onclose?.();
      fixture.detectChanges();

      expect(api.getConsultation).toHaveBeenCalledWith('consultation-1');
      expect(component.consultationClosed).toBe(true);
      expect(component.statusMessage).toContain('conversation is read-only');
      expect(socket.close).toHaveBeenCalled();
      expect(setTimeoutSpy).not.toHaveBeenCalled();
      expect(fixture.nativeElement.textContent).toContain(
        'Consultation ended. Conversation is read-only.',
      );
      expect(fixture.nativeElement.querySelector('#patient-message')).toBeNull();
      component.messageText = 'This must not be sent.';
      await component.sendTypedMessage();
      expect(api.sendMessage).not.toHaveBeenCalled();
      expect(component.error).toContain('consultation is read-only');
    } finally {
      setTimeoutSpy.mockRestore();
    }
  });

  it('states that an interpreter request is not a booking', async () => {
    const api = {
      sendMessage: vi.fn().mockReturnValue(of({})),
      getMessages: vi.fn().mockReturnValue(of({ messages: [], total: 0, offset: 0, limit: 100 })),
      openNotifications: vi.fn().mockReturnValue({ close: vi.fn() }),
    };
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [{ provide: PatientApiService, useValue: api }],
    }).compileComponents();
    const fixture = TestBed.createComponent(App);
    const component = fixture.componentInstance;
    component.sessionId = 'session-1';
    component.conversationId = 'conversation-1';

    await component.requestInterpreter();

    expect(api.sendMessage).toHaveBeenCalledWith(
      'conversation-1',
      'session-1',
      'I would like to request an interpreter.',
      'TEXT',
    );
    expect(component.statusMessage).toContain('does not book an interpreter');
  });
});
