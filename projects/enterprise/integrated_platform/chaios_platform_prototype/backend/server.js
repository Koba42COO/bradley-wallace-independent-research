
const express = require('express');
const cors = require('cors');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

app.get('/api/health', (req, res) => {
    res.json({ status: 'healthy', timestamp: new Date().toISOString() });
});

app.get('/api/data', (req, res) => {
    res.json({ message: 'chAIos API is running', data: [] });
});

app.listen(PORT, () => {
    console.log(`chAIos API server running on port ${PORT}`);
});
