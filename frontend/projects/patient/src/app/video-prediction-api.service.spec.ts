import {
  HttpErrorResponse,
  HttpEventType,
  HttpResponse,
  provideHttpClient,
} from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';

import { VideoPredictionApiService } from './video-prediction-api.service';

describe('VideoPredictionApiService', () => {
  let api: VideoPredictionApiService;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    api = TestBed.inject(VideoPredictionApiService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  it('uploads multipart video and emits progress and the prediction response', () => {
    const file = new File(['synthetic video'], 'sample.MOV', {
      type: 'application/octet-stream',
    });
    const events: unknown[] = [];
    api.predictVideo(file).subscribe((event) => events.push(event));

    const request = http.expectOne('http://localhost:8000/api/v1/predictions');
    expect(request.request.method).toBe('POST');
    expect(request.request.reportProgress).toBe(true);
    expect(request.request.headers.has('Content-Type')).toBe(false);
    const formData = request.request.body;
    expect(formData).toBeInstanceOf(FormData);
    if (!(formData instanceof FormData)) {
      throw new Error('Expected a multipart form body.');
    }
    expect(formData.get('video')).toBeInstanceOf(File);

    request.event({ type: HttpEventType.UploadProgress, loaded: 50, total: 100 });
    const response = {
      predictionStatus: 'UNCONFIRMED' as const,
      modelVersion: 'bilstm-v1',
      scoreType: 'uncalibrated_softmax' as const,
      frameCount: 12,
      candidates: [{ label: 'SIGN-1', score: 0.7 }],
      warning: 'Unconfirmed',
    };
    request.flush(response);

    expect(events).toHaveLength(3);
    expect(events[0]).toMatchObject({ type: HttpEventType.Sent });
    expect(events[1]).toMatchObject({
      type: HttpEventType.UploadProgress,
      loaded: 50,
      total: 100,
    });
    expect(events[2]).toBeInstanceOf(HttpResponse);
    expect(events[2]).toMatchObject({ body: response });
  });

  it('emits HTTP response errors', () => {
    const errors: unknown[] = [];
    api.predictVideo(new File(['video'], 'sample.mp4')).subscribe({
      error: (error: unknown) => errors.push(error),
    });
    const request = http.expectOne('http://localhost:8000/api/v1/predictions');
    request.flush({ detail: 'invalid video' }, { status: 422, statusText: 'Unprocessable Entity' });

    expect(errors).toHaveLength(1);
    expect(errors[0]).toBeInstanceOf(HttpErrorResponse);
    expect((errors[0] as HttpErrorResponse).status).toBe(422);
  });

  it('emits network errors', () => {
    const errors: unknown[] = [];
    api.predictVideo(new File(['video'], 'sample.mp4')).subscribe({
      error: (error: unknown) => errors.push(error),
    });
    const request = http.expectOne('http://localhost:8000/api/v1/predictions');
    request.error(new ProgressEvent('error'));

    expect(errors).toHaveLength(1);
  });
});
