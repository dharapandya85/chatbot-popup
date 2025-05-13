const OpenAI = require("openai");
const fs = require("fs").promises;
const path = require("path");

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const EMBEDDING_MODEL = "text-embedding-3-small";
const CHAT_MODEL = "gpt-4o";

function cosineSimilarity(vecA, vecB) {
  const dot = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
  const magA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
  const magB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
  return dot / (magA * magB);
}

async function chatWithRAG(userQuery) {
  const vectorStorePath = path.join(process.cwd(), "public/embeddings/vector_store.json");
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
        content: "You are an Agent AI,introduce yourself as Aryan AI Agentbot.Also tell about your features. Use this context:\n\n" + context,
      },
      { role: "user", content: userQuery },
    ],
  });

  return completion.choices[0].message.content;
}

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Only POST allowed" });
  }

  try {
    const { message } = req.body;
    const reply = await chatWithRAG(message);
    res.status(200).json({ reply });
  } catch (err) {
    console.error(err);
    res.status(500).json({ reply: "Server error." });
  }
}