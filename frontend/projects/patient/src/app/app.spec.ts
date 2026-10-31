import { TestBed } from '@angular/core/testing';
import { App } from './app';

describe('Patient application shell', () => {
  it('states that the prototype does not yet recognize ISL', async () => {
    await TestBed.configureTestingModule({ imports: [App] }).compileComponents();
    const fixture = TestBed.createComponent(App);
    fixture.detectChanges();

    expect(fixture.nativeElement.textContent).toContain(
      'This application does not recognize Indian Sign Language yet.',
    );
    expect(fixture.nativeElement.textContent).toContain('No consultation is connected.');
  });
});
