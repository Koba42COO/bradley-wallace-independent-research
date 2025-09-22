import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule, ReactiveFormsModule, FormBuilder, FormGroup, Validators } from '@angular/forms';
import { IonicModule } from '@ionic/angular';
import { RouterModule, Router } from '@angular/router';
import { ApiService } from '../../core/services/api';

export interface User {
  id: string;
  username: string;
  email: string;
  full_name: string;
  role: string;
  created_at: string;
  last_login: string;
  is_active: boolean;
}

export interface SystemStats {
  users: {
    total: number;
    active: number;
    new_today: number;
    roles: Record<string, number>;
  };
  system: {
    uptime: number;
    cpu_usage: number;
    memory_usage: number;
    disk_usage: number;
    requests_today: number;
    errors_today: number;
  };
  consciousness: {
    processing_requests: number;
    average_response_time: number;
    success_rate: number;
    consciousness_score: number;
    active_sessions: number;
  };
  performance: {
    api_response_time: number;
    database_connections: number;
    cache_hit_rate: number;
    websocket_connections: number;
  };
}

@Component({
  selector: 'app-admin',
  templateUrl: './admin.page.html',
  styleUrls: ['./admin.page.scss'],
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    ReactiveFormsModule,
    IonicModule,
    RouterModule
  ],
})
export class AdminPage implements OnInit {
  users: User[] = [];
  systemStats: SystemStats | null = null;
  loading = false;
  selectedTab = 'dashboard';
  userForm: FormGroup;
  showUserModal = false;

  constructor(
    private apiService: ApiService,
    private formBuilder: FormBuilder,
    public router: Router
  ) {
    this.userForm = this.formBuilder.group({
      username: ['', [Validators.required, Validators.minLength(3)]],
      email: ['', [Validators.required, Validators.email]],
      full_name: ['', [Validators.required]],
      password: ['', [Validators.required, Validators.minLength(6)]],
      role: ['user', [Validators.required]]
    });
  }

  ngOnInit() {
    this.checkAdminAccess();
    this.loadDashboardData();
  }

  checkAdminAccess() {
    const userData = localStorage.getItem('user_data');
    if (userData) {
      const user = JSON.parse(userData);
      if (user.role !== 'admin') {
        this.router.navigate(['/home']);
        return;
      }
    } else {
      this.router.navigate(['/login']);
      return;
    }
  }

  loadDashboardData() {
    this.loading = true;
    
    // Load users and system stats in parallel
    Promise.all([
      this.loadUsers(),
      this.loadSystemStats()
    ]).finally(() => {
      this.loading = false;
    });
  }

  async loadUsers() {
    try {
      const response = await this.apiService.getAdminUsers().toPromise();
      if (response?.success) {
        this.users = response.data;
      }
    } catch (error) {
      console.error('Failed to load users:', error);
    }
  }

  async loadSystemStats() {
    try {
      const response = await this.apiService.getAdminSystemStats().toPromise();
      if (response?.success) {
        this.systemStats = response.data;
      }
    } catch (error) {
      console.error('Failed to load system stats:', error);
    }
  }

  onTabChange(tab: string) {
    this.selectedTab = tab;
  }

  openUserModal() {
    this.showUserModal = true;
    this.userForm.reset();
    this.userForm.patchValue({ role: 'user' });
  }

  closeUserModal() {
    this.showUserModal = false;
  }

  async createUser() {
    if (this.userForm.valid) {
      try {
        const userData = this.userForm.value;
        const response = await this.apiService.createUserAdmin(userData).toPromise();
        
        if (response?.success) {
          this.closeUserModal();
          this.loadUsers(); // Refresh user list
        }
      } catch (error) {
        console.error('Failed to create user:', error);
      }
    }
  }

  async deleteUser(userId: string) {
    if (confirm('Are you sure you want to delete this user?')) {
      try {
        const response = await this.apiService.deleteUserAdmin(userId).toPromise();
        if (response?.success) {
          this.loadUsers(); // Refresh user list
        }
      } catch (error) {
        console.error('Failed to delete user:', error);
      }
    }
  }

  formatUptime(seconds: number): string {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
  }

  getRoleColor(role: string): string {
    switch (role) {
      case 'admin': return 'danger';
      case 'researcher': return 'warning';
      case 'user': return 'primary';
      default: return 'medium';
    }
  }

  getStatusColor(value: number, thresholds = { warning: 70, danger: 90 }): string {
    if (value >= thresholds.danger) return 'danger';
    if (value >= thresholds.warning) return 'warning';
    return 'success';
  }
}
