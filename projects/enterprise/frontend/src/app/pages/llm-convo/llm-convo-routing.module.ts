import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';

import { LlmConvoPage } from './llm-convo.page';

const routes: Routes = [
  {
    path: '',
    component: LlmConvoPage
  }
];

@NgModule({
  imports: [RouterModule.forChild(routes)],
  exports: [RouterModule],
})
export class LlmConvoPageRoutingModule {}
