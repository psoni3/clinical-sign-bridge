import { HttpErrorResponse, HttpEvent, HttpEventType, HttpResponse } from '@angular/common/http';
import { TestBed } from '@angular/core/testing';
import { of, Subject, throwError } from 'rxjs';

import { App } from './app';
import { PatientApiService } from './patient-api.service';
import { VideoPredictionApiService, VideoPredictionResponse } from './video-prediction-api.service';

function fileSelectionEvent(input: HTMLInputElement): Event {
  const event = new Event('change');
  Object.defineProperty(event, 'target', { value: input });
  return event;
}

describe('Patient application', () => {
  it('shows the no-auth prototype notice and consultation join form', async () => {
    const api = {
      joinConsultation: vi.fn(),
      getConsultation: vi.fn(),
    };
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [
        { provide: PatientApiService, useValue: api },
        { provide: VideoPredictionApiService, useValue: { predictVideo: vi.fn() } },
      ],
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
      providers: [
        { provide: PatientApiService, useValue: api },
        { provide: VideoPredictionApiService, useValue: { predictVideo: vi.fn() } },
      ],
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
      providers: [
        { provide: PatientApiService, useValue: api },
        { provide: VideoPredictionApiService, useValue: { predictVideo: vi.fn() } },
      ],
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
      providers: [
        { provide: PatientApiService, useValue: api },
        { provide: VideoPredictionApiService, useValue: { predictVideo: vi.fn() } },
      ],
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
      providers: [
        { provide: PatientApiService, useValue: api },
        { provide: VideoPredictionApiService, useValue: { predictVideo: vi.fn() } },
      ],
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
      providers: [
        { provide: PatientApiService, useValue: api },
        { provide: VideoPredictionApiService, useValue: { predictVideo: vi.fn() } },
      ],
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
      providers: [
        { provide: PatientApiService, useValue: api },
        { provide: VideoPredictionApiService, useValue: { predictVideo: vi.fn() } },
      ],
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

  it('validates MP4/MOV file extensions and upload size without relying on MIME type', async () => {
    const predictionApi = { predictVideo: vi.fn() };
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [
        { provide: PatientApiService, useValue: {} },
        { provide: VideoPredictionApiService, useValue: predictionApi },
      ],
    }).compileComponents();
    const fixture = TestBed.createComponent(App);
    const component = fixture.componentInstance;

    const movInput = document.createElement('input');
    movInput.type = 'file';
    Object.defineProperty(movInput, 'files', {
      value: [new File(['video'], 'sample.MOV', { type: 'application/octet-stream' })],
    });
    component.onPredictionFileSelected(fileSelectionEvent(movInput));

    expect(component.predictionFile?.name).toBe('sample.MOV');
    expect(component.predictionError).toBe('');

    const unsupportedInput = document.createElement('input');
    unsupportedInput.type = 'file';
    Object.defineProperty(unsupportedInput, 'files', {
      value: [new File(['video'], 'sample.webm', { type: 'video/webm' })],
    });
    component.onPredictionFileSelected(fileSelectionEvent(unsupportedInput));

    expect(component.predictionFile).toBeNull();
    expect(component.predictionError).toContain('MP4 or MOV');

    const oversizedFile = new File(['video'], 'large.mp4');
    Object.defineProperty(oversizedFile, 'size', { value: 50 * 1024 * 1024 + 1 });
    const oversizedInput = document.createElement('input');
    oversizedInput.type = 'file';
    Object.defineProperty(oversizedInput, 'files', { value: [oversizedFile] });
    component.onPredictionFileSelected(fileSelectionEvent(oversizedInput));

    expect(component.predictionFile).toBeNull();
    expect(component.predictionError).toContain('50 MiB');
    expect(predictionApi.predictVideo).not.toHaveBeenCalled();
  });

  it('displays five unconfirmed candidates and keeps patient selection local', async () => {
    const candidates = Array.from({ length: 5 }, (_, index) => ({
      label: `SIGN-${index + 1}`,
      score: 0.5 - index * 0.05,
    }));
    const predictionEvents = new Subject<HttpEvent<VideoPredictionResponse>>();
    const predictionApi = { predictVideo: vi.fn().mockReturnValue(predictionEvents) };
    const patientApi = { confirmInterpretation: vi.fn(), sendMessage: vi.fn() };
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [
        { provide: PatientApiService, useValue: patientApi },
        { provide: VideoPredictionApiService, useValue: predictionApi },
      ],
    }).compileComponents();
    const fixture = TestBed.createComponent(App);
    const component = fixture.componentInstance;
    component.predictionFile = new File(['video'], 'sample.mp4');

    fixture.detectChanges();
    component.uploadPrediction();
    predictionEvents.next({
      type: HttpEventType.UploadProgress,
      loaded: 50,
      total: 100,
    });
    fixture.detectChanges();

    expect(component.predictionProgress).toBe(50);
    expect(fixture.nativeElement.textContent).toContain('50%');

    predictionEvents.next(
      new HttpResponse({
        body: {
          predictionStatus: 'UNCONFIRMED',
          modelVersion: 'bilstm-v1',
          scoreType: 'uncalibrated_softmax',
          frameCount: 12,
          candidates,
          warning: 'AI-generated candidates are unconfirmed.',
        },
      }),
    );
    predictionEvents.complete();
    fixture.detectChanges();

    expect(predictionApi.predictVideo).toHaveBeenCalledWith(component.predictionFile);
    expect(component.prediction?.candidates).toHaveLength(5);
    expect(fixture.nativeElement.textContent).toContain('UNCONFIRMED');
    expect(fixture.nativeElement.textContent).toContain('Uncalibrated score: 0.5000');
    expect(fixture.nativeElement.textContent).toContain('SIGN-5');
    expect(component.predictionProgress).toBe(100);

    component.selectPredictionCandidate(candidates[0]);

    expect(component.selectedPredictionLabel).toBe('SIGN-1');
    expect(component.selectedCandidateId).toBeNull();
    expect(component.confirmedText).toBe('');
    expect(patientApi.confirmInterpretation).not.toHaveBeenCalled();
    expect(patientApi.sendMessage).not.toHaveBeenCalled();
    expect(fixture.nativeElement.textContent).toContain(
      'Selecting a candidate here does not confirm it',
    );
  });

  it('shows a safe error when prediction upload fails', async () => {
    const predictionApi = {
      predictVideo: vi.fn().mockReturnValue(throwError(() => new HttpErrorResponse({ status: 0 }))),
    };
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [
        { provide: PatientApiService, useValue: {} },
        { provide: VideoPredictionApiService, useValue: predictionApi },
      ],
    }).compileComponents();
    const component = TestBed.createComponent(App).componentInstance;
    component.predictionFile = new File(['video'], 'sample.mp4');

    component.uploadPrediction();

    expect(component.predictionLoading).toBe(false);
    expect(component.predictionError).toContain('could not be reached');
    expect(component.prediction).toBeNull();
  });

  it('maps server validation errors to safe user-facing messages', async () => {
    const predictionApi = {
      predictVideo: vi.fn().mockReturnValue(
        throwError(
          () =>
            new HttpErrorResponse({
              status: 422,
              error: { detail: 'private server path' },
            }),
        ),
      ),
    };
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [
        { provide: PatientApiService, useValue: {} },
        { provide: VideoPredictionApiService, useValue: predictionApi },
      ],
    }).compileComponents();
    const component = TestBed.createComponent(App).componentInstance;
    component.predictionFile = new File(['video'], 'sample.mp4');

    component.uploadPrediction();

    expect(component.predictionError).toContain('could not be processed');
    expect(component.predictionError).not.toContain('private');
  });
});
