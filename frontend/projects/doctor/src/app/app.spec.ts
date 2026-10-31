import { TestBed } from '@angular/core/testing';
import { App } from './app';

describe('Doctor application shell', () => {
  it('keeps unverified suggestions distinct from confirmed messages', async () => {
    await TestBed.configureTestingModule({ imports: [App] }).compileComponents();
    const fixture = TestBed.createComponent(App);
    fixture.detectChanges();

    expect(fixture.nativeElement.textContent).toContain('No consultations are available.');
    expect(fixture.nativeElement.textContent).toContain(
      'Unverified AI suggestions are not displayed as patient-confirmed messages.',
    );
  });
});
