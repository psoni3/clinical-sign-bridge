import { HttpErrorResponse } from '@angular/common/http';
import { TestBed } from '@angular/core/testing';
import { of, Subject, throwError } from 'rxjs';

import { App } from './app';
import { PatientApiService, RecognitionDto } from './patient-api.service';

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
      providers: [{ provide: PatientApiService, useValue: api }],
    }).compileComponents();
    const fixture = TestBed.createComponent(App);
    fixture.detectChanges();

    expect(fixture.nativeElement.textContent).toContain(
      'This application does not recognize Indian Sign Language.',
    );
    expect(fixture.nativeElement.textContent).toContain('Join a consultation');
    expect(fixture.nativeElement.textContent).toContain('No account or password is needed.');
    expect(fixture.nativeElement.textContent).toContain(
      'This temporary session and any in-progress recognition are not restored after a page refresh.',
    );
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

  it('sends a correction through the existing correction API only', async () => {
    const api = {
      correctInterpretation: vi.fn().mockReturnValue(of({})),
      getMessages: vi.fn().mockReturnValue(of({ messages: [], total: 0, offset: 0, limit: 100 })),
      sendMessage: vi.fn(),
    };
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [{ provide: PatientApiService, useValue: api }],
    }).compileComponents();
    const component = TestBed.createComponent(App).componentInstance;
    component.sessionId = 'session-1';
    component.conversationId = 'conversation-1';
    component.interpretation = {
      interpretationId: 'interpretation-1',
      status: 'UNVERIFIED',
      modelVersion: 'mock-v1',
      quality: { status: 'ACCEPTABLE', messages: [] },
      candidates: [
        { candidateId: 'candidate-1', rank: 1, text: 'Original text', confidence: null },
      ],
      availableActions: ['CORRECT'],
    };
    component.selectedCandidateId = 'candidate-1';
    component.correctionText = 'Corrected text';

    await component.correctSelectedCandidate();

    expect(api.correctInterpretation).toHaveBeenCalledWith(
      'interpretation-1',
      'candidate-1',
      'Corrected text',
      'session-1',
      expect.any(String),
    );
    expect(api.sendMessage).not.toHaveBeenCalled();
    expect(component.confirmedText).toBe('Corrected text');
  });

  it('rejects candidates without creating a patient message', async () => {
    const api = {
      rejectInterpretation: vi.fn().mockReturnValue(of({})),
      getMessages: vi.fn().mockReturnValue(of({ messages: [], total: 0, offset: 0, limit: 100 })),
      sendMessage: vi.fn(),
    };
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [{ provide: PatientApiService, useValue: api }],
    }).compileComponents();
    const component = TestBed.createComponent(App).componentInstance;
    component.sessionId = 'session-1';
    component.conversationId = 'conversation-1';
    component.interpretation = {
      interpretationId: 'interpretation-1',
      status: 'UNVERIFIED',
      modelVersion: 'mock-v1',
      quality: { status: 'NEEDS_CLARIFICATION', messages: [] },
      candidates: [],
      availableActions: ['REJECT_ALL'],
    };

    await component.rejectCandidates();

    expect(api.rejectInterpretation).toHaveBeenCalledWith('interpretation-1', 'session-1');
    expect(api.sendMessage).not.toHaveBeenCalled();
    expect(component.statusMessage).toContain('No patient-confirmed message was created');
  });

  it('starts another local signing attempt without submitting recognition', async () => {
    const api = { uploadSegment: vi.fn() };
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [{ provide: PatientApiService, useValue: api }],
    }).compileComponents();
    const fixture = TestBed.createComponent(App);
    const component = fixture.componentInstance;
    component.stage = 'review';
    component.cameraAvailable = true;
    component.selectedCandidateId = 'candidate-1';
    component.interpretation = {
      interpretationId: 'interpretation-1',
      status: 'UNVERIFIED',
      modelVersion: 'mock-v1',
      quality: { status: 'ACCEPTABLE', messages: [] },
      candidates: [{ candidateId: 'candidate-1', rank: 1, text: 'Example', confidence: null }],
      availableActions: ['CONFIRM'],
    };
    const input = document.createElement('input');
    input.type = 'file';
    Object.defineProperty(input, 'files', {
      value: [new File(['video'], 'sample.mp4', { type: 'video/mp4' })],
    });
    component.onRecognitionFileSelected(fileSelectionEvent(input));
    fixture.detectChanges();

    component.trySigningAgain();

    expect(component.stage).toBe('capture');
    expect(component.interpretation).toBeNull();
    expect(component.recognitionFile).toBeNull();
    expect(component.statusMessage).toContain('No new recognition has been submitted');
    expect(fixture.nativeElement.textContent).toContain('Start another signing attempt');
    expect(api.uploadSegment).not.toHaveBeenCalled();
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
    component.stage = 'conversation';
    component.consentAccepted = false;
    component.liveConsentRecorded = false;
    component.messageText = 'Please repeat the question.';
    fixture.detectChanges();

    expect(fixture.nativeElement.querySelector('#patient-message')).not.toBeNull();
    expect(api.sendMessage).not.toHaveBeenCalled();
    await component.sendTypedMessage();

    expect(api.sendMessage).toHaveBeenCalledWith(
      'conversation-1',
      'session-1',
      'Please repeat the question.',
      'TEXT',
    );
    expect(component.stage).toBe('conversation');
  });

  it('does not submit camera WebM and offers the MP4/MOV and typed alternatives', async () => {
    const api = {
      uploadSegment: vi.fn(),
      getMessages: vi.fn().mockReturnValue(of({ messages: [], total: 0, offset: 0, limit: 100 })),
      openNotifications: vi.fn().mockReturnValue({ close: vi.fn() }),
    };
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [{ provide: PatientApiService, useValue: api }],
    }).compileComponents();
    const fixture = TestBed.createComponent(App);
    const component = fixture.componentInstance;
    component.stage = 'capture';
    component.recordedClip = new Blob(['camera recording'], { type: 'video/webm' });
    component.consultationId = 'consultation-1';
    component.sessionId = 'session-1';
    component.liveConsentRecorded = true;
    fixture.detectChanges();

    expect(fixture.nativeElement.textContent).toContain(
      'Camera recordings are currently WebM, which sign recognition does not support.',
    );
    expect(fixture.nativeElement.textContent).toContain(
      'This WebM camera recording was not uploaded.',
    );
    const buttonLabels = Array.from(
      fixture.nativeElement.querySelectorAll('button'),
      (button: HTMLButtonElement) => button.textContent ?? '',
    );
    expect(buttonLabels.some((label) => label.includes('Upload this segment'))).toBe(false);
    expect(fixture.nativeElement.querySelector('#recognition-video')).not.toBeNull();
    expect(fixture.nativeElement.textContent).toContain('Cancel and type a message');

    const fileInput = fixture.nativeElement.querySelector('#recognition-video') as HTMLInputElement;
    const clickSpy = vi.spyOn(fileInput, 'click');
    const buttons = Array.from(
      fixture.nativeElement.querySelectorAll('button') as NodeListOf<HTMLButtonElement>,
    );
    const chooseFileButton = buttons.find((button) =>
      button.textContent?.includes('Choose an existing MP4/MOV'),
    );
    expect(chooseFileButton).toBeDefined();
    expect(chooseFileButton?.disabled).toBe(false);
    chooseFileButton?.click();
    expect(clickSpy).toHaveBeenCalledOnce();

    await component.uploadSegment();
    await component.uploadSegment();

    expect(api.uploadSegment).not.toHaveBeenCalled();
    expect(component.stage).toBe('capture');
    expect(component.interpretation).toBeNull();
    expect(component.busy).toBe(false);
    expect(component.error).toContain('recordings are WebM');
    expect(component.statusMessage).toBe('');
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

  it('rejects unsupported extensions, MIME types, and oversized files before upload', async () => {
    const api = { uploadSegment: vi.fn() };
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [{ provide: PatientApiService, useValue: api }],
    }).compileComponents();
    const fixture = TestBed.createComponent(App);
    const component = fixture.componentInstance;

    const webmInput = document.createElement('input');
    webmInput.type = 'file';
    Object.defineProperty(webmInput, 'files', {
      value: [new File(['video'], 'sample.webm', { type: 'video/webm' })],
    });
    component.onRecognitionFileSelected(fileSelectionEvent(webmInput));
    component.consultationId = 'consultation-1';
    component.sessionId = 'session-1';
    component.liveConsentRecorded = true;
    await component.uploadRecognitionFile();

    expect(component.recognitionFile).toBeNull();
    expect(component.recognitionFileError).toContain('MP4 or MOV');
    expect(api.uploadSegment).not.toHaveBeenCalled();

    const invalidMimeInput = document.createElement('input');
    invalidMimeInput.type = 'file';
    Object.defineProperty(invalidMimeInput, 'files', {
      value: [new File(['video'], 'sample.mp4', { type: 'video/quicktime' })],
    });
    component.onRecognitionFileSelected(fileSelectionEvent(invalidMimeInput));
    expect(component.recognitionFile).toBeNull();
    expect(component.recognitionFileError).toContain('video/mp4');

    const oversizedFile = new File(['video'], 'large.mp4');
    Object.defineProperty(oversizedFile, 'size', { value: 50 * 1024 * 1024 + 1 });
    const oversizedInput = document.createElement('input');
    oversizedInput.type = 'file';
    Object.defineProperty(oversizedInput, 'files', { value: [oversizedFile] });
    Object.defineProperty(oversizedFile, 'type', { value: 'video/mp4' });
    component.onRecognitionFileSelected(fileSelectionEvent(oversizedInput));

    expect(component.recognitionFile).toBeNull();
    expect(component.recognitionFileError).toContain('50 MiB');
    expect(api.uploadSegment).not.toHaveBeenCalled();
  });

  it('requires an active patient session and consent before uploading a selected file', async () => {
    const api = { uploadSegment: vi.fn() };
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [{ provide: PatientApiService, useValue: api }],
    }).compileComponents();
    const component = TestBed.createComponent(App).componentInstance;
    const input = document.createElement('input');
    input.type = 'file';
    Object.defineProperty(input, 'files', {
      value: [new File(['video'], 'sample.mp4', { type: 'video/mp4' })],
    });
    component.onRecognitionFileSelected(fileSelectionEvent(input));

    await component.uploadRecognitionFile();

    expect(api.uploadSegment).not.toHaveBeenCalled();
    expect(component.recognitionFileError).toContain('Join a consultation');
  });

  it('clears an expired session and guides the patient to rejoin', async () => {
    const api = {
      uploadSegment: vi.fn().mockReturnValue(
        throwError(
          () =>
            new HttpErrorResponse({
              status: 401,
              error: { detail: { code: 'SESSION_EXPIRED' } },
            }),
        ),
      ),
    };
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [{ provide: PatientApiService, useValue: api }],
    }).compileComponents();
    const component = TestBed.createComponent(App).componentInstance;
    component.consultationId = 'consultation-1';
    component.joinCode = 'ABCDEF012345';
    component.sessionId = 'expired-session';
    component.conversationId = 'conversation-1';
    component.liveConsentRecorded = true;
    const input = document.createElement('input');
    input.type = 'file';
    Object.defineProperty(input, 'files', {
      value: [new File(['video'], 'sample.mp4', { type: 'video/mp4' })],
    });
    component.onRecognitionFileSelected(fileSelectionEvent(input));

    await component.uploadRecognitionFile();

    expect(component.stage).toBe('join');
    expect(component.sessionId).toBe('');
    expect(component.conversationId).toBe('');
    expect(component.joinCode).toBe('');
    expect(component.error).toContain('session is no longer valid');
    expect(component.statusMessage).toContain('Rejoin with the consultation ID and join code');
    expect(component.consultationId).toBe('consultation-1');
    await component.uploadRecognitionFile();
    expect(api.uploadSegment).toHaveBeenCalledTimes(1);
  });

  it.each([
    { name: 'sample.mp4', type: 'video/mp4' },
    { name: 'sample.mov', type: 'video/quicktime' },
  ])(
    'uploads $name through recognition and confirms only after explicit action',
    async ({ name, type }) => {
      const interpretation = {
        interpretationId: 'interpretation-1',
        status: 'UNVERIFIED' as const,
        modelVersion: 'bilstm-v1',
        quality: { status: 'ACCEPTABLE', messages: [] },
        candidates: [
          { candidateId: 'candidate-1', rank: 1, text: 'Example phrase', confidence: null },
          { candidateId: 'candidate-2', rank: 2, text: 'Alternative phrase', confidence: null },
        ],
        availableActions: ['CONFIRM', 'CORRECT', 'REJECT_ALL'],
      };
      const api = {
        uploadSegment: vi.fn().mockReturnValue(
          of({
            recognitionId: 'recognition-1',
            status: 'COMPLETED',
            interpretationId: 'interpretation-1',
          }),
        ),
        getInterpretation: vi.fn().mockReturnValue(of(interpretation)),
        confirmInterpretation: vi.fn().mockReturnValue(of({})),
        getMessages: vi.fn().mockReturnValue(of({ messages: [], total: 0, offset: 0, limit: 100 })),
        sendMessage: vi.fn(),
      };
      await TestBed.configureTestingModule({
        imports: [App],
        providers: [{ provide: PatientApiService, useValue: api }],
      }).compileComponents();
      const fixture = TestBed.createComponent(App);
      const component = fixture.componentInstance;
      component.consultationId = 'consultation-1';
      component.sessionId = 'session-1';
      component.liveConsentRecorded = true;
      const file = new File(['synthetic video'], name, { type });
      const input = document.createElement('input');
      input.type = 'file';
      Object.defineProperty(input, 'files', { value: [file] });
      component.onRecognitionFileSelected(fileSelectionEvent(input));

      await component.uploadRecognitionFile();
      fixture.detectChanges();
      const requestId = api.uploadSegment.mock.calls[0][2];
      expect(api.uploadSegment).toHaveBeenCalledWith(
        'consultation-1',
        'session-1',
        requestId,
        file,
        name,
      );
      expect(api.getInterpretation).toHaveBeenCalledWith('interpretation-1', 'session-1');
      expect(component.interpretation?.candidates).toHaveLength(2);
      expect(component.stage).toBe('review');
      expect(fixture.nativeElement.textContent).toContain('Example phrase');
      const candidateRadio = fixture.nativeElement.querySelector(
        'input[name="candidate"]',
      ) as HTMLInputElement;
      candidateRadio.dispatchEvent(new Event('change'));
      fixture.detectChanges();
      expect(component.selectedCandidateId).toBe('candidate-1');
      expect(api.confirmInterpretation).not.toHaveBeenCalled();
      expect(api.sendMessage).not.toHaveBeenCalled();

      await component.confirmSelectedCandidate();

      expect(api.confirmInterpretation).toHaveBeenCalledWith(
        'interpretation-1',
        'candidate-1',
        'session-1',
        expect.any(String),
      );
      expect(component.stage).toBe('conversation');
      expect(api.sendMessage).not.toHaveBeenCalled();
    },
  );

  it('polls the recognition and retrieves the persisted interpretation', async () => {
    const api = {
      uploadSegment: vi
        .fn()
        .mockReturnValue(
          of({ recognitionId: 'recognition-1', status: 'PROCESSING', interpretationId: null }),
        ),
      getRecognition: vi.fn().mockReturnValue(
        of({
          recognitionId: 'recognition-1',
          status: 'COMPLETED',
          interpretationId: 'interpretation-1',
        }),
      ),
      getInterpretation: vi.fn().mockReturnValue(
        of({
          interpretationId: 'interpretation-1',
          status: 'UNVERIFIED',
          modelVersion: 'bilstm-v1',
          quality: { status: 'ACCEPTABLE', messages: [] },
          candidates: [],
          availableActions: [],
        }),
      ),
    };
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [{ provide: PatientApiService, useValue: api }],
    }).compileComponents();
    const component = TestBed.createComponent(App).componentInstance;
    component.consultationId = 'consultation-1';
    component.sessionId = 'session-1';
    component.liveConsentRecorded = true;
    const input = document.createElement('input');
    input.type = 'file';
    Object.defineProperty(input, 'files', {
      value: [new File(['video'], 'sample.mp4', { type: 'video/mp4' })],
    });
    component.onRecognitionFileSelected(fileSelectionEvent(input));

    await component.uploadRecognitionFile();

    expect(api.getRecognition).toHaveBeenCalledWith('recognition-1', 'session-1');
    expect(api.getInterpretation).toHaveBeenCalledWith('interpretation-1', 'session-1');
    expect(component.interpretation?.interpretationId).toBe('interpretation-1');
  });

  it('resumes timed-out polling without submitting the recognition again', async () => {
    const pending: RecognitionDto = {
      recognitionId: 'recognition-1',
      status: 'PROCESSING',
      interpretationId: null,
    };
    const completed: RecognitionDto = {
      recognitionId: 'recognition-1',
      status: 'COMPLETED',
      interpretationId: 'interpretation-1',
    };
    const getRecognition = vi
      .fn()
      .mockReturnValueOnce(of(pending))
      .mockReturnValueOnce(of(pending))
      .mockReturnValueOnce(of(pending))
      .mockReturnValueOnce(of(pending))
      .mockReturnValueOnce(of(pending))
      .mockReturnValueOnce(of(pending))
      .mockReturnValueOnce(of(pending))
      .mockReturnValueOnce(of(pending))
      .mockReturnValueOnce(of(pending))
      .mockReturnValueOnce(of(pending))
      .mockReturnValueOnce(of(pending))
      .mockReturnValueOnce(of(pending))
      .mockReturnValueOnce(of(completed));
    const api = {
      uploadSegment: vi.fn().mockReturnValue(of(pending)),
      getRecognition,
      getInterpretation: vi.fn().mockReturnValue(
        of({
          interpretationId: 'interpretation-1',
          status: 'UNVERIFIED',
          modelVersion: 'mock-v1',
          quality: { status: 'ACCEPTABLE', messages: [] },
          candidates: [],
          availableActions: [],
        }),
      ),
    };
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [{ provide: PatientApiService, useValue: api }],
    }).compileComponents();
    const component = TestBed.createComponent(App).componentInstance;
    component.consultationId = 'consultation-1';
    component.sessionId = 'session-1';
    component.liveConsentRecorded = true;
    const input = document.createElement('input');
    input.type = 'file';
    Object.defineProperty(input, 'files', {
      value: [new File(['video'], 'sample.mp4', { type: 'video/mp4' })],
    });
    component.onRecognitionFileSelected(fileSelectionEvent(input));

    vi.useFakeTimers();
    try {
      const firstCheck = component.uploadRecognitionFile();
      await vi.runAllTimersAsync();
      await firstCheck;

      expect(component.error).toContain('still processing');
      expect(component.recognitionSubmitLabel).toBe('Check recognition status');
      expect(api.uploadSegment).toHaveBeenCalledTimes(1);

      await component.uploadRecognitionFile();

      expect(api.uploadSegment).toHaveBeenCalledTimes(1);
      expect(getRecognition).toHaveBeenCalledTimes(13);
      expect(component.stage).toBe('review');
    } finally {
      vi.useRealTimers();
    }
  });

  it('resumes interrupted polling without a second upload', async () => {
    const api = {
      uploadSegment: vi
        .fn()
        .mockReturnValue(
          of({ recognitionId: 'recognition-1', status: 'PROCESSING', interpretationId: null }),
        ),
      getRecognition: vi
        .fn()
        .mockReturnValueOnce(
          throwError(
            () =>
              new HttpErrorResponse({
                status: 503,
                error: { detail: { message: 'Status service unavailable.' } },
              }),
          ),
        )
        .mockReturnValueOnce(
          of({
            recognitionId: 'recognition-1',
            status: 'COMPLETED',
            interpretationId: 'interpretation-1',
          }),
        ),
      getInterpretation: vi.fn().mockReturnValue(
        of({
          interpretationId: 'interpretation-1',
          status: 'UNVERIFIED',
          modelVersion: 'mock-v1',
          quality: { status: 'ACCEPTABLE', messages: [] },
          candidates: [],
          availableActions: [],
        }),
      ),
    };
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [{ provide: PatientApiService, useValue: api }],
    }).compileComponents();
    const component = TestBed.createComponent(App).componentInstance;
    component.consultationId = 'consultation-1';
    component.sessionId = 'session-1';
    component.liveConsentRecorded = true;
    const input = document.createElement('input');
    input.type = 'file';
    Object.defineProperty(input, 'files', {
      value: [new File(['video'], 'sample.mp4', { type: 'video/mp4' })],
    });
    component.onRecognitionFileSelected(fileSelectionEvent(input));

    vi.useFakeTimers();
    try {
      const firstCheck = component.uploadRecognitionFile();
      await vi.runAllTimersAsync();
      await firstCheck;

      expect(component.error).toContain('status checking was interrupted');
      expect(component.error).toContain('without uploading again');

      await component.uploadRecognitionFile();

      expect(api.uploadSegment).toHaveBeenCalledTimes(1);
      expect(api.getRecognition).toHaveBeenCalledTimes(2);
      expect(component.stage).toBe('review');
    } finally {
      vi.useRealTimers();
    }
  });

  it('reuses the request ID after a failed upload of the unchanged file', async () => {
    const api = {
      uploadSegment: vi
        .fn()
        .mockReturnValueOnce(
          throwError(
            () =>
              new HttpErrorResponse({
                status: 503,
                error: { detail: { message: 'Inference is temporarily unavailable.' } },
              }),
          ),
        )
        .mockReturnValueOnce(
          of({
            recognitionId: 'recognition-1',
            status: 'COMPLETED',
            interpretationId: 'interpretation-1',
          }),
        ),
      getInterpretation: vi.fn().mockReturnValue(
        of({
          interpretationId: 'interpretation-1',
          status: 'UNVERIFIED',
          modelVersion: 'bilstm-v1',
          quality: { status: 'ACCEPTABLE', messages: [] },
          candidates: [],
          availableActions: [],
        }),
      ),
    };
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [{ provide: PatientApiService, useValue: api }],
    }).compileComponents();
    const component = TestBed.createComponent(App).componentInstance;
    component.consultationId = 'consultation-1';
    component.sessionId = 'session-1';
    component.liveConsentRecorded = true;
    const input = document.createElement('input');
    input.type = 'file';
    Object.defineProperty(input, 'files', {
      value: [new File(['video'], 'sample.mp4', { type: 'video/mp4' })],
    });
    component.onRecognitionFileSelected(fileSelectionEvent(input));

    await component.uploadRecognitionFile();
    const firstRequestId = api.uploadSegment.mock.calls[0][2];
    expect(component.error).toContain('Inference is temporarily unavailable.');

    await component.uploadRecognitionFile();

    expect(api.uploadSegment).toHaveBeenCalledTimes(2);
    expect(api.uploadSegment.mock.calls[1][2]).toBe(firstRequestId);
  });

  it('creates a new request ID when a different file is selected', async () => {
    const api = {
      uploadSegment: vi
        .fn()
        .mockReturnValueOnce(
          throwError(
            () =>
              new HttpErrorResponse({
                status: 503,
                error: { detail: { message: 'Temporary upload failure.' } },
              }),
          ),
        )
        .mockReturnValueOnce(
          of({
            recognitionId: 'recognition-2',
            status: 'COMPLETED',
            interpretationId: 'interpretation-2',
          }),
        ),
      getInterpretation: vi.fn().mockReturnValue(
        of({
          interpretationId: 'interpretation-2',
          status: 'UNVERIFIED',
          modelVersion: 'mock-v1',
          quality: { status: 'ACCEPTABLE', messages: [] },
          candidates: [],
          availableActions: [],
        }),
      ),
    };
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [{ provide: PatientApiService, useValue: api }],
    }).compileComponents();
    const component = TestBed.createComponent(App).componentInstance;
    component.consultationId = 'consultation-1';
    component.sessionId = 'session-1';
    component.liveConsentRecorded = true;
    const selectFile = (file: File): void => {
      const input = document.createElement('input');
      input.type = 'file';
      Object.defineProperty(input, 'files', { value: [file] });
      component.onRecognitionFileSelected(fileSelectionEvent(input));
    };

    selectFile(new File(['first'], 'first.mp4', { type: 'video/mp4' }));
    await component.uploadRecognitionFile();
    const firstRequestId = api.uploadSegment.mock.calls[0][2];
    selectFile(new File(['second'], 'second.mp4', { type: 'video/mp4' }));
    await component.uploadRecognitionFile();

    expect(api.uploadSegment).toHaveBeenCalledTimes(2);
    expect(api.uploadSegment.mock.calls[1][2]).not.toBe(firstRequestId);
  });

  it('retries a failed recognition with the same request ID', async () => {
    const api = {
      uploadSegment: vi
        .fn()
        .mockReturnValueOnce(
          of({ recognitionId: 'recognition-1', status: 'FAILED', interpretationId: null }),
        )
        .mockReturnValueOnce(
          of({
            recognitionId: 'recognition-1',
            status: 'COMPLETED',
            interpretationId: 'interpretation-1',
          }),
        ),
      getInterpretation: vi.fn().mockReturnValue(
        of({
          interpretationId: 'interpretation-1',
          status: 'UNVERIFIED',
          modelVersion: 'mock-v1',
          quality: { status: 'ACCEPTABLE', messages: [] },
          candidates: [],
          availableActions: [],
        }),
      ),
    };
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [{ provide: PatientApiService, useValue: api }],
    }).compileComponents();
    const component = TestBed.createComponent(App).componentInstance;
    component.consultationId = 'consultation-1';
    component.sessionId = 'session-1';
    component.liveConsentRecorded = true;
    const input = document.createElement('input');
    input.type = 'file';
    Object.defineProperty(input, 'files', {
      value: [new File(['video'], 'sample.mp4', { type: 'video/mp4' })],
    });
    component.onRecognitionFileSelected(fileSelectionEvent(input));

    await component.uploadRecognitionFile();
    const requestId = api.uploadSegment.mock.calls[0][2];
    expect(component.error).toContain('Recognition failed');
    await component.uploadRecognitionFile();

    expect(api.uploadSegment).toHaveBeenCalledTimes(2);
    expect(api.uploadSegment.mock.calls[1][2]).toBe(requestId);
  });

  it('prevents a second submission while the first upload is pending', async () => {
    const uploadResult = new Subject<RecognitionDto>();
    const api = {
      uploadSegment: vi.fn().mockReturnValue(uploadResult),
      getInterpretation: vi.fn().mockReturnValue(
        of({
          interpretationId: 'interpretation-1',
          status: 'UNVERIFIED',
          modelVersion: 'bilstm-v1',
          quality: { status: 'ACCEPTABLE', messages: [] },
          candidates: [],
          availableActions: [],
        }),
      ),
    };
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [{ provide: PatientApiService, useValue: api }],
    }).compileComponents();
    const component = TestBed.createComponent(App).componentInstance;
    component.consultationId = 'consultation-1';
    component.sessionId = 'session-1';
    component.liveConsentRecorded = true;
    const input = document.createElement('input');
    input.type = 'file';
    Object.defineProperty(input, 'files', {
      value: [new File(['video'], 'sample.mp4', { type: 'video/mp4' })],
    });
    component.onRecognitionFileSelected(fileSelectionEvent(input));

    const firstSubmission = component.uploadRecognitionFile();
    await component.uploadRecognitionFile();
    expect(api.uploadSegment).toHaveBeenCalledTimes(1);

    uploadResult.next({
      recognitionId: 'recognition-1',
      status: 'COMPLETED',
      interpretationId: 'interpretation-1',
    });
    uploadResult.complete();
    await firstSubmission;
  });
});
