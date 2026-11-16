import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';

import { PATIENT_API_BASE_URL, PatientApiService } from './patient-api.service';

describe('PatientApiService', () => {
  let api: PatientApiService;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
        { provide: PATIENT_API_BASE_URL, useValue: 'http://localhost:8000/api/v1' },
      ],
    });
    api = TestBed.inject(PatientApiService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  it('joins using the camelCase M1 request contract', () => {
    api.joinConsultation('consultation-1', 'ABCDEF012345').subscribe();
    const request = http.expectOne(
      'http://localhost:8000/api/v1/consultations/consultation-1/join',
    );

    expect(request.request.method).toBe('POST');
    expect(request.request.body).toEqual({
      joinCode: 'ABCDEF012345',
      deviceLabel: 'Patient device',
      sessionType: 'PATIENT_DEVICE',
    });
    request.flush({
      consultationId: 'consultation-1',
      sessionId: 'session-1',
      sessionToken: 'temporary-token',
      status: 'JOINED',
    });
  });

  it.each([
    { name: 'sample.mp4', type: 'video/mp4' },
    { name: 'sample.mov', type: 'video/quicktime' },
  ])('uploads $name with its filename, MIME type, session, and request ID', ({ name, type }) => {
    const video = new File(['synthetic video'], name, { type });
    api.uploadSegment('consultation-1', 'session-1', 'request-1', video, video.name).subscribe();
    const request = http.expectOne(
      'http://localhost:8000/api/v1/consultations/consultation-1/recognitions',
    );
    const form = request.request.body as FormData;
    const uploadedVideo = form.get('video');

    expect(request.request.method).toBe('POST');
    expect(form.get('clientRequestId')).toBe('request-1');
    expect(form.get('sessionId')).toBe('session-1');
    expect(uploadedVideo).toBeInstanceOf(File);
    if (!(uploadedVideo instanceof File)) {
      throw new Error('Expected the multipart video to be a File.');
    }
    expect(uploadedVideo.name).toBe(name);
    expect(uploadedVideo.type).toBe(type);
    request.flush({
      recognitionId: 'recognition-1',
      status: 'COMPLETED',
      interpretationId: 'interpretation-1',
    });
  });

  it('preserves the existing camera multipart filename when no filename is supplied', () => {
    const segment = new Blob(['synthetic segment'], { type: 'video/webm' });
    api.uploadSegment('consultation-1', 'session-1', 'request-1', segment).subscribe();
    const request = http.expectOne(
      'http://localhost:8000/api/v1/consultations/consultation-1/recognitions',
    );
    const form = request.request.body as FormData;
    const uploadedVideo = form.get('video');

    expect(request.request.method).toBe('POST');
    expect(form.get('clientRequestId')).toBe('request-1');
    expect(form.get('sessionId')).toBe('session-1');
    expect(uploadedVideo).toBeInstanceOf(File);
    if (!(uploadedVideo instanceof File)) {
      throw new Error('Expected the multipart video to be a File.');
    }
    expect(uploadedVideo.name).toBe('signing-segment.webm');
    expect(uploadedVideo.type).toBe('video/webm');
    request.flush({
      recognitionId: 'recognition-1',
      status: 'UNSUPPORTED',
      interpretationId: 'interpretation-1',
    });
  });

  it('fetches conversation history with a bound session and page size', () => {
    api.getMessages('conversation-1', 'session-1').subscribe();
    const request = http.expectOne(
      (pending) =>
        pending.url === 'http://localhost:8000/api/v1/conversations/conversation-1/messages' &&
        pending.params.get('sessionId') === 'session-1' &&
        pending.params.get('limit') === '100',
    );

    expect(request.request.method).toBe('GET');
    request.flush({ messages: [], total: 0, offset: 0, limit: 100 });
  });
});
