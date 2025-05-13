const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const fs = require('fs').promises;
const path = require('path');
const OpenAI = require('openai');

dotenv.config();

const app = express();
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

const EMBEDDING_MODEL = "text-embedding-3-small";
const CHAT_MODEL = "gpt-4o";

// --- Cosine Similarity Helper ---
function cosineSimilarity(vecA, vecB) {
  const dot = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
  const magA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
  const magB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
  return dot / (magA * magB);
}

// --- RAG Chat Handler ---
async function chatWithRAG(userQuery) {
  const vectorStorePath=path.join(__dirname,'public/embeddings/vector_store.json');
  const vectorStore = JSON.parse(await fs.readFile(vectorStorePath, "utf8"));
  const queryEmbedding = (
    await openai.embeddings.create({ model: EMBEDDING_MODEL, input: userQuery })
  ).data[0].embedding;

  const topChunks = vectorStore
    .map((item) => ({
      text: item.text,
      score: cosineSimilarity(queryEmbedding, item.embedding),
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 3);

  const context = topChunks.map((c) => c.text).join("\n");

  const completion = await openai.chat.completions.create({
    model: CHAT_MODEL,
    messages: [
      {
        role: "system",
        content: "You are an assistant for Base blockchain. Use this context:\n\n" + context,
      },
      { role: "user", content: userQuery },
    ],
  });

  return completion.choices[0].message.content;
}

// --- Chat API ---
app.post('/api/chat', async (req, res) => {
  try {
    const userMessage = req.body.message;
    const reply = await chatWithRAG(userMessage);
    res.json({ reply });
  } catch (err) {
    console.error(err);
    res.status(500).json({ reply: "Server error." });
  }
});

// --- Optional: Embed data manually ---
app.get('/embed', async (req, res) => {
  const rawPath=path.join(__dirname,'public/data/base_knowledge.txt');
  const raw = await fs.readFile(rawPath, "utf8");
  const chunks = raw.split("\n").filter(Boolean);
  const embeddings = await Promise.all(
    chunks.map(async (text) => {
      const res = await openai.embeddings.create({
        model: EMBEDDING_MODEL,
        input: text,
      });
      return { text, embedding: res.data[0].embedding };
    })
  );
  const vectorStorePath=path.join(__dirname,'public/embeddings/vector_store.json');
  await fs.writeFile(vectorStorePath, JSON.stringify(embeddings, null, 2));
  res.send("Embeddings saved.");
});

// --- Start Server ---
const PORT = 3000;
app.listen(PORT, () => console.log(`Server running at http://localhost:${PORT}`));