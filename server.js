import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import { CohereClient } from "cohere-ai";

const app = express();
app.use(cors());
app.use(express.json());

// Setup Cohere client
const cohere = new CohereClient({
  token: process.env.COHERE_API_KEY  // move token to .env for safety
});

// POST /generate — stream chat response back
app.post('/generate', async (req, res) => {
  const { prompt } = req.body;

  if (!prompt) {
    return res.status(400).json({ error: 'Prompt is required' });
  }

  try {
    // Start chat stream
    const stream = await cohere.chatStream({
      model: "be8fa8b0-162a-47b6-b20a-e9aded1601be-ft", // your fine-tuned model
      message: prompt
    });

    // We'll accumulate the stream output
    let fullText = "";

    for await (const chat of stream) {
      if (chat.eventType === "text-generation") {
        fullText += chat.text;
      }
    }

    // Respond with the full generated text
    res.json({ text: fullText.trim() });

  } catch (err) {
    console.error('Error communicating with Cohere API:', err);
    res.status(500).json({ error: 'Cohere request failed' });
  }
});

app.listen(5000, () => {
  console.log('Listening on http://localhost:5000');
});
