const express = require('express');
const { body, validationResult } = require('express-validator');
const Data = require('../models/Data');
const { protect } = require('../middleware/auth');

const router = express.Router();

// @route   GET /api/data
// @desc    Get all data items
// @access  Private
router.get('/', protect, async (req, res) => {
  try {
    const { category, search, page = 1, limit = 10 } = req.query;
    const query = {};

    // Filter by category
    if (category) {
      query.category = category;
    }

    // Filter by public or user's own data
    query.$or = [
      { isPublic: true },
      { createdBy: req.user._id }
    ];

    // Text search
    if (search) {
      query.$text = { $search: search };
    }

    const skip = (parseInt(page) - 1) * parseInt(limit);

    const data = await Data.find(query)
      .populate('createdBy', 'username email')
      .sort({ createdAt: -1 })
      .skip(skip)
      .limit(parseInt(limit));

    const total = await Data.countDocuments(query);

    res.json({
      success: true,
      count: data.length,
      total,
      page: parseInt(page),
      pages: Math.ceil(total / parseInt(limit)),
      data
    });
  } catch (error) {
    res.status(500).json({
      message: 'Error fetching data',
      error: error.message
    });
  }
});

// @route   GET /api/data/:id
// @desc    Get data item by ID
// @access  Private
router.get('/:id', protect, async (req, res) => {
  try {
    const data = await Data.findById(req.params.id)
      .populate('createdBy', 'username email');

    if (!data) {
      return res.status(404).json({
        message: 'Data item not found'
      });
    }

    // Check if user has access
    if (!data.isPublic && data.createdBy._id.toString() !== req.user._id.toString()) {
      return res.status(403).json({
        message: 'Not authorized to view this data'
      });
    }

    res.json({
      success: true,
      data
    });
  } catch (error) {
    res.status(500).json({
      message: 'Error fetching data',
      error: error.message
    });
  }
});

// @route   POST /api/data
// @desc    Create new data item
// @access  Private
router.post('/', protect, [
  body('title').trim().notEmpty().withMessage('Title is required'),
  body('content').notEmpty().withMessage('Content is required'),
  body('category').optional().isIn(['general', 'research', 'documentation', 'code', 'other']),
  body('tags').optional().isArray()
], async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ errors: errors.array() });
    }

    const data = await Data.create({
      ...req.body,
      createdBy: req.user._id
    });

    await data.populate('createdBy', 'username email');

    res.status(201).json({
      success: true,
      data
    });
  } catch (error) {
    res.status(500).json({
      message: 'Error creating data',
      error: error.message
    });
  }
});

// @route   PUT /api/data/:id
// @desc    Update data item
// @access  Private
router.put('/:id', protect, [
  body('title').optional().trim().notEmpty(),
  body('category').optional().isIn(['general', 'research', 'documentation', 'code', 'other'])
], async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ errors: errors.array() });
    }

    let data = await Data.findById(req.params.id);

    if (!data) {
      return res.status(404).json({
        message: 'Data item not found'
      });
    }

    // Check if user owns this data
    if (data.createdBy.toString() !== req.user._id.toString()) {
      return res.status(403).json({
        message: 'Not authorized to update this data'
      });
    }

    data = await Data.findByIdAndUpdate(
      req.params.id,
      req.body,
      { new: true, runValidators: true }
    ).populate('createdBy', 'username email');

    res.json({
      success: true,
      data
    });
  } catch (error) {
    res.status(500).json({
      message: 'Error updating data',
      error: error.message
    });
  }
});

// @route   DELETE /api/data/:id
// @desc    Delete data item
// @access  Private
router.delete('/:id', protect, async (req, res) => {
  try {
    const data = await Data.findById(req.params.id);

    if (!data) {
      return res.status(404).json({
        message: 'Data item not found'
      });
    }

    // Check if user owns this data
    if (data.createdBy.toString() !== req.user._id.toString()) {
      return res.status(403).json({
        message: 'Not authorized to delete this data'
      });
    }

    await Data.findByIdAndDelete(req.params.id);

    res.json({
      success: true,
      message: 'Data item deleted successfully'
    });
  } catch (error) {
    res.status(500).json({
      message: 'Error deleting data',
      error: error.message
    });
  }
});

module.exports = router;

