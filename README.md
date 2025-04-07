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
const teapot = await TeapotAI.fromPretrained({
  llmId: "teapotai/teapotllm",
  embeddingModelId: "tomasmcm/teapotai-teapotembedding-onnx",
  documents: ["Some context document", "Another document"],
  settings: {
    useRag: true,
    ragNumResults: 3,
    ragSimilarityThreshold: 0.5,
    maxContextLength: 512,
    contextChunking: true,
    verbose: true,
    logLevel: "info"
  },
  dtype: "q4",
  device: "webgpu"
});
```

## API Reference

### TeapotAI Class

#### Static Methods

- `fromPretrained(options?)`: Creates a new TeapotAI instance with the specified model
  - `options`: Configuration options including:
    - `llmId`: Optional model ID (default: "teapotai/teapotllm")
    - `embeddingModelId`: Optional embedding model ID (default: "tomasmcm/teapotai-teapotembedding-onnx")
    - `documents`: Array of documents for RAG
    - `settings`: TeapotAI settings
    - `dtype`: Model data type (e.g., "q4", "fp16")
    - `device`: Execution device (e.g., "webgpu", "cpu")
    - `progressCallback`: Function to monitor loading progress

#### Instance Methods

- `query(query, context?, systemPrompt?)`: Generate a response for a given query
  - `query`: The question to answer
  - `context`: Optional context information (string)
  - `systemPrompt`: Optional system prompt to use

- `chat(conversationHistory)`: Process a conversation and generate a response
  - `conversationHistory`: Array of message objects with `role` and `content`

- `extract(schema, query?, context?)`: Extract structured data based on a schema
  - `schema`: Object defining the data structure to extract
  - `query`: Optional query to provide context for extraction
  - `context`: Optional additional context

- `rag(query)`: Perform Retrieval-Augmented Generation
  - `query`: The query to find relevant documents for
  - Returns relevant document chunks

- `generate(inputText)`: Direct text generation using the model
  - `inputText`: Raw input text to pass to the model
  - Returns generated text output

### Settings

The `settings` object in the configuration options supports the following properties:

- `useRag` (boolean): Enable Retrieval Augmented Generation
- `ragNumResults` (number): Maximum number of documents to retrieve in RAG
- `ragSimilarityThreshold` (number): Minimum similarity threshold for RAG results
- `maxContextLength` (number): Maximum context length for generation
- `contextChunking` (boolean): Enable automatic chunking of large documents
- `verbose` (boolean): Enable verbose logging
- `logLevel` (string): Log level ("info", "debug", etc.)

## Examples

### Using Context for Accurate Answers

```javascript
const context = `
  The Eiffel Tower is a wrought iron lattice tower in Paris, France. 
  It was designed by Gustave Eiffel and completed in 1889.
  It stands at a height of 330 meters and is one of the most recognizable structures in the world.
`;

const response = await teapot.query("What is the height of the Eiffel Tower?", context);
// Output: The Eiffel Tower stands at a height of 330 meters.
```

### Information Extraction

```javascript
const apartmentDescription = `
  This spacious 2-bedroom apartment is available for rent in downtown New York. 
  The monthly rent is $2500. It includes 1 bathroom and a fully equipped kitchen.
  Pets are welcome! Please reach out to us at 555-123-4567.
`;

const schema = {
  rent: { type: 'number', description: 'the monthly rent in dollars' },
  bedrooms: { type: 'number', description: 'the number of bedrooms' },
  bathrooms: { type: 'number', description: 'the number of bathrooms' },
  phone_number: { type: 'string' }
};

const extractedInfo = await teapot.extract(schema, "", apartmentDescription);
// Output: { rent: 2500, bedrooms: 2, bathrooms: 1, phone_number: '555-123-4567' }
```

### Using RAG for Knowledge Retrieval

```javascript
const documents = [
  "The Eiffel Tower is located in Paris, France. It was built in 1889 and stands 330 meters tall.",
  "The Great Wall of China is a historic fortification that stretches over 13,000 miles.",
  "The Empire State Building is an iconic skyscraper in New York City that was completed in 1931."
];

const teapot = await TeapotAI.fromPretrained(undefined, {
  documents,
  settings: { useRag: true }
});

const response = await teapot.chat([
  { role: "system", content: "You are an agent designed to answer facts about famous landmarks." },
  { role: "user", content: "What landmark was constructed in the 1800s?" }
]);
// Output: The Eiffel Tower was constructed in 1889.
```

## Browser Usage

For directly using TeapotAI.js in the browser with a script tag:

```html
<script src="https://cdn.jsdelivr.net/npm/teapotai-js/dist/teapotai.web.js"></script>
<script>
  async function runTeapot() {
    const teapot = await TeapotAI.fromPretrained();
    const answer = await teapot.query("Hello, how are you?");
    document.getElementById("answer").textContent = answer;
  }
  runTeapot();
</script>
```

## License

MIT License
