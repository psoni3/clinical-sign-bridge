import { HttpClient, HttpEvent } from '@angular/common/http';
import { inject, Injectable } from '@angular/core';
import { Observable } from 'rxjs';

import { environment } from '../environments/environment';

export interface VideoPredictionCandidate {
  label: string;
  score: number;
}

export interface VideoPredictionResponse {
  predictionStatus: 'UNCONFIRMED';
  modelVersion: string;
  scoreType: 'uncalibrated_softmax';
  frameCount: number;
  candidates: VideoPredictionCandidate[];
  warning: string;
}

@Injectable({ providedIn: 'root' })
export class VideoPredictionApiService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = environment.aiPredictionApiBaseUrl.replace(/\/$/, '');

  predictVideo(file: File): Observable<HttpEvent<VideoPredictionResponse>> {
    const form = new FormData();
    form.append('video', file, file.name);

    return this.http.post<VideoPredictionResponse>(`${this.baseUrl}/predictions`, form, {
      reportProgress: true,
      observe: 'events',
    });
  }
}
