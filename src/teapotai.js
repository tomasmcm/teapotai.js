import { pipeline, env } from "@huggingface/transformers";

const DEFAULT_MODEL = "tomasmcm/teapotai-teapotllm-onnx";
const DEFAULT_SYSTEM_PROMPT = "You are Teapot, an open-source AI assistant optimized for low-end devices, providing short, accurate responses without hallucinating while excelling at information extraction and text summarization. If a user asks who you are reply \"I am Teapot\". When a user says 'you' they mean 'Teapot', so answer question from the perspective of Teapot.";

env.cacheDir = './.cache';

/**
 * @typedef {Object} TeapotAISettings
 * @property {boolean} [useRag=true] Whether to use Retrieval Augmented Generation
 * @property {number} [ragNumResults=3] Number of top documents to retrieve based on similarity
 * @property {number} [ragSimilarityThreshold=0.5] Similarity threshold for document relevance
 * @property {number} [maxContextLength=512] Maximum length of context to consider
 * @property {boolean} [contextChunking=true] Whether to chunk context for processing
 * @property {boolean} [verbose=false] Whether to print verbose updates
 * @property {string} [logLevel="info"] Log level setting (e.g., 'info', 'debug')
 */

/**
 * @typedef {Object} ConversationMessage
 * @property {string} role The role of the message sender (e.g., 'user', 'assistant')
 * @property {string} content The content of the message
 */

export class TeapotAI {
  /**
   * Create a new TeapotAI instance.
   * @param {Object} model The model
   * @param {string[]} [documents=[]] List of documents to use for context retrieval
   * @param {TeapotAISettings} [settings={}] Settings for TeapotAI
   */
  constructor(model, documents = [], settings = {}) {
    this.model = model;
    this.documents = [];
    
    // Apply default settings and override with provided settings
    this.settings = {
      useRag: false,
      ragNumResults: 3,
      ragSimilarityThreshold: 0.3,
      maxContextLength: 512,
      contextChunking: true,
      verbose: false,
      logLevel: "info",
      ...settings
    };

    if (this.settings.verbose) {
      console.log(`
 _____                      _         _    ___        __o__    _;;
|_   _|__  __ _ _ __   ___ | |_      / \\  |_ _|   __ /-___-\\__/ /
  | |/ _ \\/ _\` | '_ \\ / _ \\| __|    / _ \\  | |   (  |       |__/
  | |  __/ (_| | |_) | (_) | |_    / ___ \\ | |    \\_|~~~~~~~|
  |_|\\___|\\_,_| .__/ \\___/ \\__/  /_/   \\_\\___|      \\_____/
               |_|   
`);
    }

    // Process and chunk the documents
    if (documents && documents.length > 0) {
      this.documents = documents.flatMap(doc => this._chunkDocument(doc));
      
      if (this.settings.useRag) {
        this._initializeEmbeddings();
      }
    }
  }

  /**
   * Load a TeapotAI model from the Hugging Face Hub.
   * @param {string} [modelId=DEFAULT_MODEL] The model id
   * @param {Object} [options={}] Additional options
   * @param {string[]} [options.documents=[]] List of documents to use for context retrieval
   * @param {TeapotAISettings} [options.settings={}] Settings for TeapotAI
   * @param {"fp32"|"fp16"|"q8"|"q4"|"q4f16"} [options.dtype="fp32"] The data type to use.
   * @param {"wasm"|"webgpu"|"cpu"|null} [options.device=null] The device to run the model on.
   * @param {Function} [options.progressCallback=null] A callback function for progress information.
   * @returns {Promise<TeapotAI>} The loaded model
   */
  static async fromPretrained(
    modelId = DEFAULT_MODEL, 
    { 
      documents = [], 
      settings = {}, 
      dtype = "q4", 
      device = null, 
      progressCallback = null 
    } = {}
  ) {
    if (settings.verbose) {
      console.log("Loading model...");
    }

    // Load the text generation pipeline
    const model = await pipeline("text2text-generation", modelId, { 
      dtype, 
      device,
      // progress_callback: progressCallback 
    });

    // Initialize the embedding model if RAG is enabled
    const teapotAI = new TeapotAI(model, documents, settings);
    
    if (settings.verbose) {
      console.log("TeapotAI initialized successfully!");
    }
    
    return teapotAI;
  }

  /**
   * Initialize the embedding pipeline for RAG functionality
   * @private
   */
  async _initializeEmbeddings() {
    if (this.settings.verbose) {
      console.log("Initializing embedding model...");
    }
    
    this.embeddingModel = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
    
    if (this.documents.length > 0) {
      this.documentEmbeddings = await this._generateDocumentEmbeddings(this.documents);
    }
    
    if (this.settings.verbose) {
      console.log("Embedding model ready!");
    }
  }

  /**
   * Split a document into smaller chunks if needed
   * @private
   * @param {string} context - The document to chunk
   * @returns {string[]} List of document chunks
   */
  _chunkDocument(context) {
    if (!this.settings.contextChunking) {
      return [context];
    }

    const tokenized = this.model.tokenizer(context);
    const maxLength = 512;

    if (tokenized.input_ids.length <= maxLength) {
      return [context];
    }

    // Chunk by paragraphs
    const paragraphs = context.split('\n\n');
    const documents = [];

    for (const paragraph of paragraphs) {
      const tokens = this.model.tokenizer(paragraph).input_ids;
      
      if (tokens.length > maxLength) {
        // Further chunk long paragraphs
        for (let i = 0; i < tokens.length; i += maxLength) {
          const chunkTokens = tokens.slice(i, i + maxLength);
          const chunkText = this.model.tokenizer.decode(chunkTokens, { skip_special_tokens: true });
          documents.push(chunkText);
        }
      } else {
        documents.push(paragraph);
      }
    }
    
    return documents;
  }

  /**
   * Generate embeddings for a list of documents
   * @private
   * @param {string[]} documents - The documents to embed
   * @returns {Promise<Array<number[]>>} The document embeddings
   */
  async _generateDocumentEmbeddings(documents) {
    if (this.settings.verbose) {
      console.log(`Generating embeddings for ${documents.length} documents...`);
    }
    
    const embeddings = [];
    
    for (const doc of documents) {
      const embedding = await this.embeddingModel(doc, { pooling: 'mean', normalize: true });
      // The embedding is returned as a Float32Array in the first position
      embeddings.push(Array.from(embedding[0]));
    }
    
    return embeddings;
  }

  /**
   * Calculate cosine similarity between two vectors
   * @private
   * @param {number[]} vecA - First vector
   * @param {number[]} vecB - Second vector
   * @returns {number} Similarity score between 0 and 1
   */
  _cosineSimilarity(vecA, vecB) {
    let dotProduct = 0.0;
    let normA = 0.0;
    let normB = 0.0;
    
    // Check if vectors are valid
    if (!Array.isArray(vecA) || !Array.isArray(vecB) || vecA.length !== vecB.length) {
      console.error('Invalid vectors for cosine similarity calculation');
      return 0;
    }

    for (let i = 0; i < vecA.length; i++) {
      // Handle potential NaN or undefined values
      const a = Number(vecA[i]) || 0;
      const b = Number(vecB[i]) || 0;
      
      dotProduct += a * b;
      normA += a * a;
      normB += b * b;
    }
    
    // Prevent division by zero
    const denominator = Math.sqrt(normA) * Math.sqrt(normB);
    if (denominator === 0) {
      return 0;
    }
    
    const similarity = dotProduct / denominator;
    
    // Ensure the result is within [-1, 1] range
    return Math.min(Math.max(similarity, -1), 1);
  }

  /**
   * Retrieve relevant documents based on a query
   * @private
   * @param {string} query - The query to retrieve documents for
   * @param {string[]} documents - The documents to search
   * @param {Array<number[]>} documentEmbeddings - The embeddings for each document
   * @returns {Promise<string[]>} List of relevant documents
   */
  async _retrieval(query, documents, documentEmbeddings) {
    const queryEmbedding = await this.embeddingModel(query, { pooling: 'mean', normalize: true });
    
    // The embedding is returned as a Float32Array in the first position
    const flatQueryEmbedding = Array.from(queryEmbedding[0]);
    
    // Calculate similarity for each document
    const similarities = documentEmbeddings.map(docEmbedding => {
      return this._cosineSimilarity(flatQueryEmbedding, docEmbedding);
    });
    
    // Filter by threshold and get indices
    const filteredIndices = similarities
      .map((similarity, index) => ({ similarity, index }))
      .filter(item => item.similarity >= this.settings.ragSimilarityThreshold)
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, this.settings.ragNumResults)
      .map(item => item.index);
    
    // Return the top documents
    return filteredIndices.map(index => documents[index]);
  }

  /**
   * Perform Retrieval-Augmented Generation (RAG) on a query
   * @param {string} query - The query to retrieve context for
   * @returns {Promise<string[]>} List of relevant document chunks
   */
  async rag(query) {
    if (!this.settings.useRag || !this.documents.length) {
      return [];
    }
    
    if (!this.embeddingModel) {
      await this._initializeEmbeddings();
    }
    
    return this._retrieval(query, this.documents, this.documentEmbeddings);
  }

  /**
   * Generate text using the model
   * @param {string} inputText - The text input to generate from
   * @returns {Promise<string>} The generated text
   */
  async generate(inputText) {
    const output = await this.model(inputText, { 
      max_new_tokens: this.settings.maxContextLength 
    });
    
    const result = output[0]?.generated_text.trim() || "";
    
    if (this.settings.logLevel === "debug") {
      console.log("Input:", inputText);
      console.log("Output:", result);
    }
    
    return result;
  }

  /**
   * Query the model with a question and optional context
   * @param {string} query - The question to answer
   * @param {string} [context=""] - Optional context to help answer the question
   * @param {string} [systemPrompt=DEFAULT_SYSTEM_PROMPT] - The system prompt to use
   * @returns {Promise<string>} The generated answer
   */
  async query(query, context = "", systemPrompt = DEFAULT_SYSTEM_PROMPT) {
    let fullContext = context;
    
    if (this.settings.useRag) {
      const ragContext = await this.rag(query);
      fullContext = ragContext.join('\n\n') + (context ? `\n${context}` : '');
    }
    
    if (this.settings.contextChunking && context) {
      const documents = this._chunkDocument(context);
      if (documents.length > this.settings.ragNumResults) {
        const documentEmbeddings = await this._generateDocumentEmbeddings(documents);
        const ragDocuments = await this._retrieval(query, documents, documentEmbeddings);
        fullContext = fullContext + '\n\n' + ragDocuments.join('\n\n');
      }
    }
    
    const inputText = `${fullContext}\n${systemPrompt}\n${query}`;
    
    return this.generate(inputText);
  }

  /**
   * Chat with the model using a conversation history
   * @param {ConversationMessage[]} conversationHistory - List of previous messages
   * @returns {Promise<string>} The model's response
   */
  async chat(conversationHistory) {
    // Find the last user message
    const lastUserIndex = [...conversationHistory]
      .reverse()
      .findIndex(msg => msg.role === 'user');
    
    if (lastUserIndex === -1) {
      return "Error: No user message found in conversation history.";
    }
    
    const adjustedIndex = conversationHistory.length - 1 - lastUserIndex;
    const lastUserMessage = conversationHistory[adjustedIndex];
    
    // Create history without the last user message
    const historyWithoutLastUser = [
      ...conversationHistory.slice(0, adjustedIndex),
      ...conversationHistory.slice(adjustedIndex + 1)
    ];
    
    const chatHistory = historyWithoutLastUser
      .map(msg => `${msg.role}: ${msg.content}`)
      .join('\n');
    
    const formattedLastUser = `user: ${lastUserMessage.content}`;
    
    return this.query(formattedLastUser, chatHistory);
  }

  /**
   * Extract structured data from text based on a schema
   * @param {Object} schema - The schema defining the data structure to extract
   * @param {string} query - The query to extract data from
   * @param {string} [context=""] - Optional context to help with extraction
   * @returns {Promise<Object>} The extracted data
   */
  async extract(schema, query = "", context = "") {
    let extractionContext = context;
    
    if (this.settings.useRag) {
      const contextDocuments = await this.rag(query);
      extractionContext = contextDocuments.join('\n') + (context ? `\n${context}` : '');
    }
    
    const result = {};
    
    for (const [fieldName, fieldInfo] of Object.entries(schema)) {
      const { type, description } = fieldInfo;
      const descriptionAnnotation = description ? `(${description})` : '';
      
      const extractedValue = await this.query(
        `Extract the field ${fieldName} ${descriptionAnnotation}`,
        extractionContext
      );
      
      let parsedResult;
      
      if (type === 'boolean') {
        parsedResult = /\b(yes|true)\b/i.test(extractedValue) ? true : 
                      /\b(no|false)\b/i.test(extractedValue) ? false : null;
      } 
      else if (type === 'number') {
        const numericValue = extractedValue.replace(/[^0-9.]/g, '');
        parsedResult = numericValue ? Number(numericValue) : null;
      } 
      else {
        // Default to string
        parsedResult = extractedValue.trim();
      }
      
      result[fieldName] = parsedResult;
    }
    
    return result;
  }
}