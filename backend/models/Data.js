const mongoose = require('mongoose');

const dataSchema = new mongoose.Schema({
  title: {
    type: String,
    required: [true, 'Title is required'],
    trim: true
  },
  description: {
    type: String,
    trim: true
  },
  content: {
    type: mongoose.Schema.Types.Mixed,
    required: true
  },
  category: {
    type: String,
    enum: ['general', 'research', 'documentation', 'code', 'other'],
    default: 'general'
  },
  tags: [{
    type: String,
    trim: true
  }],
  createdBy: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  isPublic: {
    type: Boolean,
    default: false
  },
  metadata: {
    type: mongoose.Schema.Types.Mixed,
    default: {}
  }
}, {
  timestamps: true
});

// Indexes for better query performance
dataSchema.index({ title: 'text', description: 'text' });
dataSchema.index({ category: 1 });
dataSchema.index({ createdBy: 1 });
dataSchema.index({ createdAt: -1 });

module.exports = mongoose.model('Data', dataSchema);

