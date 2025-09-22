import { ComponentFixture, TestBed } from '@angular/core/testing';
import { LlmConvoPage } from './llm-convo.page';

describe('LlmConvoPage', () => {
  let component: LlmConvoPage;
  let fixture: ComponentFixture<LlmConvoPage>;

  beforeEach(() => {
    fixture = TestBed.createComponent(LlmConvoPage);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
