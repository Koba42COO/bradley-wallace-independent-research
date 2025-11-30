import { Component, OnInit } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { ActivatedRoute, Router } from '@angular/router';
import { DataService } from '../../services/data.service';

@Component({
  selector: 'app-data-form',
  templateUrl: './data-form.component.html',
  styleUrls: ['./data-form.component.css']
})
export class DataFormComponent implements OnInit {
  dataForm: FormGroup;
  isEditMode: boolean = false;
  dataId: string | null = null;
  loading: boolean = false;
  error: string = '';

  categories = ['general', 'research', 'documentation', 'code', 'other'];

  constructor(
    private fb: FormBuilder,
    private dataService: DataService,
    private route: ActivatedRoute,
    private router: Router
  ) {
    this.dataForm = this.fb.group({
      title: ['', Validators.required],
      description: [''],
      content: ['', Validators.required],
      category: ['general', Validators.required],
      tags: [''],
      isPublic: [false]
    });
  }

  ngOnInit(): void {
    this.dataId = this.route.snapshot.paramMap.get('id');
    if (this.dataId) {
      this.isEditMode = true;
      this.loadData();
    }
  }

  loadData(): void {
    if (this.dataId) {
      this.loading = true;
      this.dataService.getDataById(this.dataId).subscribe({
        next: (response) => {
          const data = response.data;
          this.dataForm.patchValue({
            title: data.title,
            description: data.description || '',
            content: typeof data.content === 'string' ? data.content : JSON.stringify(data.content),
            category: data.category,
            tags: data.tags?.join(', ') || '',
            isPublic: data.isPublic || false
          });
          this.loading = false;
        },
        error: (err) => {
          this.error = err.error?.message || 'Failed to load data';
          this.loading = false;
        }
      });
    }
  }

  onSubmit(): void {
    if (this.dataForm.valid) {
      this.loading = true;
      this.error = '';

      const formValue = this.dataForm.value;
      const data = {
        ...formValue,
        tags: formValue.tags ? formValue.tags.split(',').map((tag: string) => tag.trim()).filter((tag: string) => tag) : [],
        content: formValue.content
      };

      if (this.isEditMode && this.dataId) {
        this.dataService.updateData(this.dataId, data).subscribe({
          next: () => {
            this.router.navigate(['/data']);
          },
          error: (err) => {
            this.error = err.error?.message || 'Failed to update data';
            this.loading = false;
          }
        });
      } else {
        this.dataService.createData(data).subscribe({
          next: () => {
            this.router.navigate(['/data']);
          },
          error: (err) => {
            this.error = err.error?.message || 'Failed to create data';
            this.loading = false;
          }
        });
      }
    }
  }
}

