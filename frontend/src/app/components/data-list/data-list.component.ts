import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { DataService } from '../../services/data.service';
import { Data } from '../../models/data.model';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-data-list',
  templateUrl: './data-list.component.html',
  styleUrls: ['./data-list.component.css'],
  standalone: false
})
export class DataListComponent implements OnInit {
  dataList: Data[] = [];
  loading: boolean = false;
  error: string = '';
  currentPage: number = 1;
  totalPages: number = 1;
  searchTerm: string = '';
  selectedCategory: string = '';

  constructor(
    private dataService: DataService,
    private router: Router
  ) { }

  ngOnInit(): void {
    this.loadData();
  }

  loadData(): void {
    this.loading = true;
    this.error = '';

    const params: any = {
      page: this.currentPage,
      limit: 10
    };

    if (this.searchTerm) {
      params.search = this.searchTerm;
    }

    if (this.selectedCategory) {
      params.category = this.selectedCategory;
    }

    this.dataService.getData(params).subscribe({
      next: (response) => {
        this.dataList = response.data;
        this.totalPages = response.pages || 1;
        this.loading = false;
      },
      error: (err) => {
        this.error = err.error?.message || 'Failed to load data';
        this.loading = false;
      }
    });
  }

  onSearch(): void {
    this.currentPage = 1;
    this.loadData();
  }

  onCategoryChange(): void {
    this.currentPage = 1;
    this.loadData();
  }

  editData(id: string): void {
    this.router.navigate(['/data/edit', id]);
  }

  deleteData(id: string): void {
    if (confirm('Are you sure you want to delete this item?')) {
      this.dataService.deleteData(id).subscribe({
        next: () => {
          this.loadData();
        },
        error: (err) => {
          this.error = err.error?.message || 'Failed to delete data';
        }
      });
    }
  }

  previousPage(): void {
    if (this.currentPage > 1) {
      this.currentPage--;
      this.loadData();
    }
  }

  nextPage(): void {
    if (this.currentPage < this.totalPages) {
      this.currentPage++;
      this.loadData();
    }
  }
}
