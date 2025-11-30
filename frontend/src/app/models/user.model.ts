export interface User {
  _id?: string;
  id?: string;
  username: string;
  email: string;
  firstName?: string;
  lastName?: string;
  role?: string;
  isActive?: boolean;
  createdAt?: string;
  updatedAt?: string;
}

export interface AuthResponse {
  success: boolean;
  token: string;
  user: User;
}

