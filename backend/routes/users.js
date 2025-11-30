const express = require('express');
const { body, validationResult } = require('express-validator');
const User = require('../models/User');
const { protect, restrictTo } = require('../middleware/auth');

const router = express.Router();

// @route   GET /api/users
// @desc    Get all users
// @access  Private/Admin
router.get('/', protect, restrictTo('admin'), async (req, res) => {
  try {
    const users = await User.find().select('-password');
    res.json({
      success: true,
      count: users.length,
      users
    });
  } catch (error) {
    res.status(500).json({
      message: 'Error fetching users',
      error: error.message
    });
  }
});

// @route   GET /api/users/:id
// @desc    Get user by ID
// @access  Private
router.get('/:id', protect, async (req, res) => {
  try {
    const user = await User.findById(req.params.id).select('-password');
    
    if (!user) {
      return res.status(404).json({
        message: 'User not found'
      });
    }

    // Users can only view their own profile unless they're admin
    if (req.user.role !== 'admin' && req.user._id.toString() !== req.params.id) {
      return res.status(403).json({
        message: 'Not authorized to view this user'
      });
    }

    res.json({
      success: true,
      user
    });
  } catch (error) {
    res.status(500).json({
      message: 'Error fetching user',
      error: error.message
    });
  }
});

// @route   PUT /api/users/:id
// @desc    Update user
// @access  Private
router.put('/:id', protect, [
  body('firstName').optional().trim(),
  body('lastName').optional().trim(),
  body('email').optional().isEmail().normalizeEmail()
], async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ errors: errors.array() });
    }

    // Users can only update their own profile unless they're admin
    if (req.user.role !== 'admin' && req.user._id.toString() !== req.params.id) {
      return res.status(403).json({
        message: 'Not authorized to update this user'
      });
    }

    const user = await User.findByIdAndUpdate(
      req.params.id,
      req.body,
      { new: true, runValidators: true }
    ).select('-password');

    if (!user) {
      return res.status(404).json({
        message: 'User not found'
      });
    }

    res.json({
      success: true,
      user
    });
  } catch (error) {
    res.status(500).json({
      message: 'Error updating user',
      error: error.message
    });
  }
});

// @route   DELETE /api/users/:id
// @desc    Delete user
// @access  Private/Admin
router.delete('/:id', protect, restrictTo('admin'), async (req, res) => {
  try {
    const user = await User.findByIdAndDelete(req.params.id);

    if (!user) {
      return res.status(404).json({
        message: 'User not found'
      });
    }

    res.json({
      success: true,
      message: 'User deleted successfully'
    });
  } catch (error) {
    res.status(500).json({
      message: 'Error deleting user',
      error: error.message
    });
  }
});

module.exports = router;

