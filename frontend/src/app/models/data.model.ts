export interface Data {
  _id?: string;
  title: string;
  description?: string;
  content: any;
  category: 'general' | 'research' | 'documentation' | 'code' | 'other';
  tags?: string[];
  createdBy?: {
    _id: string;
    username: string;
    email: string;
  };
  isPublic?: boolean;
  metadata?: any;
  createdAt?: string;
  updatedAt?: string;
}

export interface DataResponse {
  success: boolean;
  count?: number;
  total?: number;
  page?: number;
  pages?: number;
  data: Data[];
}

