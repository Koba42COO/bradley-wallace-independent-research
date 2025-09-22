import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { IonicModule } from '@ionic/angular';

import { LlmConvoPageRoutingModule } from './llm-convo-routing.module';
import { LlmConvoPage } from './llm-convo.page';

@NgModule({
  imports: [
    CommonModule,
    FormsModule,
    IonicModule,
    LlmConvoPageRoutingModule,
    LlmConvoPage
  ],
  declarations: []
})
export class LlmConvoPageModule {}
