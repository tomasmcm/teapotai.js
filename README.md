# TeapotAI.js

🫖 AI powered LLM agents, privately brewed in your browser.

TeapotAI.js is a lightweight JavaScript library for running [TeapotLLM](https://huggingface.co/teapotai/teapotllm) directly in the browser or Node.js environment. It provides an easy-to-use interface for text generation, chat functionality, and information extraction capabilities.

## Features

- 🚀 Run LLMs directly in browser or Node.js
- 🔒 Private by design - all processing happens locally
- 📚 Retrieval Augmented Generation (RAG) support
- 💬 Chat interface with conversation history
- 🔍 Structured data extraction
- 🧩 Context chunking for handling long documents
- ⚡ Optimized for low-end devices

## Installation

```bash
npm install teapotai-js
```

## Quick Start

```javascript
import { TeapotAI } from 'teapotai-js';

// Initialize TeapotAI with default model
const teapot = await TeapotAI.fromPretrained();

// Simple query
const answer = await teapot.query("What is TeapotAI?");

// Chat conversation
const conversation = [
  { role: "user", content: "Hello! Who are you?" },
  { role: "assistant", content: "I am Teapot" },
  { role: "user", content: "What can you help me with?" }
];
const response = await teapot.chat(conversation);

// Extract structured data
const schema = {
  name: { type: "string", description: "Extract the person's name" },
  age: { type: "number", description: "Extract the age" },
  isStudent: { type: "boolean", description: "Is the person a student?" }
};
const data = await teapot.extract(schema, "John is a 25-year-old student");
```

## Advanced Configuration

```javascript
const teapot = await TeapotAI.fromPretrained("tomasmcm/teapotai-teapotllm-onnx", {
  documents: ["Some context document", "Another document"],
  settings: {
    useRag: true,
    ragNumResults: 3,
    ragSimilarityThreshold: 0.5,
    maxContextLength: 512,
    contextChunking: true,
    verbose: true,
    logLevel: "debug"
  },
  dtype: "q4",
  device: "webgpu"
});
```

## API Reference

### TeapotAI Class

#### Static Methods

- `fromPretrained(modelId?, options?)`: Creates a new TeapotAI instance with the specified model

#### Instance Methods

- `query(query, context?, systemPrompt?)`: Generate a response for a given query
- `chat(conversationHistory)`: Process a conversation and generate a response
- `extract(schema, query?, context?)`: Extract structured data based on a schema
- `rag(query)`: Perform Retrieval-Augmented Generation
- `generate(inputText)`: Direct text generation using the model

## License

MIT License
