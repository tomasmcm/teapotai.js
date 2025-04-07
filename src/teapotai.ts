import {
  env,
  pipeline,
} from "@huggingface/transformers";
import type {
  FeatureExtractionPipeline,
  Message,
  ProgressInfo,
  Text2TextGenerationPipeline,
} from "@huggingface/transformers";
import { cosineSimilarity } from "./cosine-similarity.js";

const DEFAULT_LLM = "teapotai/teapotllm";
const DEFAULT_EMBEDING_MODEL = "tomasmcm/teapotai-teapotembedding-onnx";
const DEFAULT_SYSTEM_PROMPT = "You are Teapot, an open-source AI assistant optimized for low-end devices, providing short, accurate responses without hallucinating while excelling at information extraction and text summarization. If a user asks who you are reply \"I am Teapot\". When a user says 'you' they mean 'Teapot', so answer question from the perspective of Teapot.";

env.cacheDir = './.cache';

/**
 * Settings for TeapotAI
 */
interface TeapotAISettings {
  useRag?: boolean;
  ragNumResults?: number;
  ragSimilarityThreshold?: number;
  maxContextLength?: number;
  contextChunking?: boolean;
  verbose?: boolean;
  logLevel?: string;
}

/**
 * Options for loading a pretrained model
 */
interface PretrainedOptions {
  llmId?: string;
  embeddingModelId?: string;
  documents?: string[];
  settings?: TeapotAISettings;
  dtype?: "q4" | "auto" | "fp32" | "fp16" | "q8" | "int8" | "uint8" | "bnb4" | "q4f16";
  device?: "auto" | "gpu" | "cpu" | "wasm" | "webgpu" | "cuda" | "dml" | "webnn" | "webnn-npu" | "webnn-gpu" | "webnn-cpu";
  progressCallback?: ((data: ProgressInfo) => void) | null;
}

/**
 * Schema field definition for data extraction
 */
interface SchemaField {
  type: string;
  description?: string;
}
interface Schema {
  [key: string]: SchemaField;
}

export class TeapotAI {
  private llm: Text2TextGenerationPipeline;
  private documents: string[];
  private settings: Required<TeapotAISettings>;
  private embeddingModel?: FeatureExtractionPipeline;
  private documentEmbeddings?: number[][];
  
  /**
   * Create a new TeapotAI instance.
   * @param model The model
   * @param documents List of documents to use for context retrieval
   * @param settings Settings for TeapotAI
   */
  constructor(
    llm: Text2TextGenerationPipeline,
    embeddingModel: FeatureExtractionPipeline,
    settings: TeapotAISettings = {}
  ) {
    this.llm = llm;
    this.embeddingModel = embeddingModel;
    this.documents = [];
    
    // Apply default settings and override with provided settings
    this.settings = {
      useRag: true,
      ragNumResults: 3,
      ragSimilarityThreshold: 0.5,
      maxContextLength: 512,
      contextChunking: true,
      verbose: true,
      logLevel: "info",
      ...settings
    };

    if (this.settings.verbose) {
      console.log(`
 _____                      _         _    ___        __o__    _;;
|_   _|__  __ _ _ __   ___ | |_      / \\  |_ _|   __ /-___-\\__/ /
  | |/ _ \\/ _\` | '_ \\ / _ \\| __|    / _ \\  | |   (  |       |__/
  | |  __/ (_| | |_) | (_) | |_    / ___ \\ | |    \\_|~~~~~~~|
  |_|\\___|\\_,__| .__/ \\___/ \\__/  /_/   \\_\\___|      \\_____/
               |_|   
`);
    }
  }

  /**
   * Load a TeapotAI model from the Hugging Face Hub.
   * @param options PretrainedOptions
   * @returns The loaded model
   */
  static async fromPretrained(
    { 
      llmId = DEFAULT_LLM,
      embeddingModelId = DEFAULT_EMBEDING_MODEL,
      documents = [], 
      settings = {}, 
      dtype = "q4", 
      device = null, 
      progressCallback = null 
    }: PretrainedOptions = {}
  ): Promise<TeapotAI> {
    if (settings.verbose) {
      console.log("Loading model...");
    }

    const teapotAI = new TeapotAI(undefined, undefined, settings);

    const [llm, embeddingModel] = await Promise.all([
      pipeline("text2text-generation", llmId, { 
        dtype, 
        device,
        progress_callback: progressCallback 
      }) as unknown as Text2TextGenerationPipeline,
      (() => {
        if (!teapotAI.settings.useRag) return null;
        return pipeline("feature-extraction", embeddingModelId, {
          dtype,
          device,
          progress_callback: progressCallback
        }) as unknown as FeatureExtractionPipeline
      })()
    ]);
    teapotAI.llm = llm;

    if (teapotAI.settings.useRag && embeddingModel) {
      teapotAI.embeddingModel = embeddingModel;
      await teapotAI.initializeEmbeddings(documents);
    }
    
    if (settings.verbose) {
      console.log("TeapotAI initialized successfully!");
    }
    
    return teapotAI;
  }

  /**
   * Initialize the embedding pipeline for RAG functionality
   */
  async initializeEmbeddings(documents): Promise<void> {
    // Process and chunk the documents
    if (documents && documents.length > 0) {
      this.documents = documents.flatMap(doc => this.chunkDocument(doc));
    }

    if (this.settings.verbose) {
      console.log("Initializing documents...");
    }
    
    this.documentEmbeddings = await this.generateDocumentEmbeddings(this.documents);
    
    if (this.settings.verbose) {
      console.log("Documents ready!");
    }
  }

  /**
   * Split a document into smaller chunks if needed
   * @param context The document to chunk
   * @returns List of document chunks
   */
  chunkDocument(context: string): string[] {
    if (!this.settings.contextChunking) {
      return [context];
    }

    const tokenized = this.llm.tokenizer(context);
    const maxLength = 512;

    if (tokenized.input_ids.length <= maxLength) {
      return [context];
    }

    // Chunk by paragraphs
    const paragraphs = context.split('\n\n');
    const documents: string[] = [];

    for (const paragraph of paragraphs) {
      const tokens = this.llm.tokenizer(paragraph).input_ids;
      
      if (tokens.length > maxLength) {
        // Further chunk long paragraphs
        for (let i = 0; i < tokens.length; i += maxLength) {
          const chunkTokens = tokens.slice(i, i + maxLength);
          const chunkText = this.llm.tokenizer.decode(chunkTokens, { skip_special_tokens: true });
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
   * @param documents The documents to embed
   * @returns The document embeddings
   */
  async generateDocumentEmbeddings(documents: string[]): Promise<number[][]> {
    if (this.settings.verbose) {
      console.log(`Generating embeddings for ${documents.length} documents...`);
    }
    
    if (!this.embeddingModel) {
      throw new Error('Embedding model not initialized');
    }
    
    const embeddings: number[][] = [];
    
    for (const doc of documents) {
      const embedding = await this.embeddingModel(doc, { pooling: 'mean', normalize: true });
      // The embedding is returned as a Float32Array in the first position
      embeddings.push(Array.from(embedding.data));
    }
    
    return embeddings;
  }

  /**
   * Retrieve relevant documents based on a query
   * @param query The query to retrieve documents for
   * @param documents The documents to search
   * @param documentEmbeddings The embeddings for each document
   * @returns List of relevant documents
   */
  async retrieval(query: string, documents: string[], documentEmbeddings: number[][]): Promise<string[]> {
    if (!this.embeddingModel) {
      throw new Error('Embedding model not initialized');
    }
    
    const queryEmbedding = await this.embeddingModel(query, { pooling: 'mean', normalize: true });
    const queryVectors = Array.from(queryEmbedding.data);
    
    // Calculate similarity for each document
    const similarities = documentEmbeddings.map(docEmbedding => {
      return cosineSimilarity(queryVectors, docEmbedding);
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
   * @param query The query to retrieve context for
   * @returns List of relevant document chunks
   */
  async rag(query: string): Promise<string[]> {
    if (!this.settings.useRag || !this.documents.length) {
      return [];
    }
    
    if (!this.documentEmbeddings || !this.embeddingModel) {
      throw new Error('Document embeddings or embedding model not initialized');
    }
    
    return this.retrieval(query, this.documents, this.documentEmbeddings);
  }

  /**
   * Generate text using the model
   * @param inputText The text input to generate from
   * @returns The generated text
   */
  async generate(inputText: string): Promise<string> {
    const output = await this.llm(inputText, { 
      max_new_tokens: this.settings.maxContextLength 
    });
    
    if (!('generated_text' in output[0])) return "Error: Could not generate text.";
    const result = output[0]?.generated_text.trim() || "";
    
    if (this.settings.logLevel === "debug") {
      console.log("Input:", inputText);
      console.log("Output:", result);
    }
    
    return result;
  }

  /**
   * Query the model with a question and optional context
   * @param query The question to answer
   * @param context Optional context to help answer the question
   * @param systemPrompt The system prompt to use
   * @returns The generated answer
   */
  async query(query: string, context: string = "", systemPrompt: string = DEFAULT_SYSTEM_PROMPT): Promise<string> {
    let fullContext = context;
    
    if (this.settings.useRag) {
      const ragContext = await this.rag(query);
      fullContext = ragContext.join('\n\n') + (context ? `\n${context}` : '');
    }
    
    if (this.settings.contextChunking && context) {
      const documents = this.chunkDocument(context);
      if (documents.length > this.settings.ragNumResults) {
        const documentEmbeddings = await this.generateDocumentEmbeddings(documents);
        const ragDocuments = await this.retrieval(query, documents, documentEmbeddings);
        fullContext = fullContext + '\n\n' + ragDocuments.join('\n\n');
      }
    }
    
    const inputText = `${fullContext}\n${systemPrompt}\n${query}`;
    
    return this.generate(inputText);
  }

  /**
   * Chat with the model using a conversation history
   * @param conversationHistory List of previous messages
   * @returns The model's response
   */
  async chat(conversationHistory: Message[]): Promise<string> {
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
   * @param schema The schema defining the data structure to extract
   * @param query The query to extract data from
   * @param context Optional context to help with extraction
   * @returns The extracted data
   */
  async extract(schema: Schema, query: string = "", context: string = ""): Promise<Record<string, any>> {
    let extractionContext = context;
    
    if (this.settings.useRag) {
      const contextDocuments = await this.rag(query);
      extractionContext = contextDocuments.join('\n') + (context ? `\n${context}` : '');
    }
    
    const result: Record<string, any> = {};
    
    for (const [fieldName, fieldInfo] of Object.entries(schema)) {
      const { type, description } = fieldInfo;
      const descriptionAnnotation = description ? `(${description})` : '';
      
      const extractedValue = await this.query(
        `Extract the field ${fieldName} ${descriptionAnnotation}`,
        extractionContext
      );
      
      let parsedResult: string | number | boolean | null;
      
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