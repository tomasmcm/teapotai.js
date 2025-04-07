// Vector store example with TeapotAI for Node.js using LangChain's MemoryVectorStore
const { TeapotAI } = require('../../dist/teapotai.cjs');
// Import MemoryVectorStore from LangChain
const { MemoryVectorStore } = require('langchain/vectorstores/memory');
// Custom embeddings class to interface with TeapotAI
class TeapotEmbeddings {
  constructor(teapotInstance) {
    this.teapot = teapotInstance;
  }

  // LangChain expects this method for generating embeddings for documents
  async embedDocuments(documents) {
    return await this.teapot.generateDocumentEmbeddings(documents);
  }
  
  // LangChain expects this method for generating query embeddings
  async embedQuery(query) {
    return (await this.teapot.generateDocumentEmbeddings([query]))[0];
  }
}

async function main() {
  try {
    console.log("Initializing TeapotAI for vector store example...");
    
    // Example documents for knowledge retrieval, same as RAG example
    const documents = [
      "The Eiffel Tower is located in Paris, France. It was built in 1889 and stands 330 meters tall.",
      "The Great Wall of China is a historic fortification that stretches over 13,000 miles.",
      "The Empire State Building is an iconic skyscraper in New York City that was completed in 1931 and stands at 1,454 feet tall.",
      "The Sahara Desert is the largest hot desert in the world, located in North Africa.",
      "The Nile River is the longest river in the world, flowing through northeastern Africa.",
      "The Amazon Rainforest is the largest tropical rainforest in the world, covering over 5.5 million square kilometers."
    ];
    
    // Initialize TeapotAI with default settings (no documents)
    const teapot = await TeapotAI.fromPretrained({
      settings: {
        useRag: true,
        contextChunking: false,
        verbose: true,
      }
    });
    
    console.log("Creating embeddings adapter for LangChain...");
    
    // Initialize the MemoryVectorStore with our TeapotAI embeddings
    console.log("Initializing MemoryVectorStore with documents...");
    const vectorStore = await MemoryVectorStore.fromTexts(
      documents,
      { id: 1, name: 'test' }, // Metadata for each document (could be more specific)
      new TeapotEmbeddings(teapot)
    );
    
    console.log(`Added ${documents.length} documents to MemoryVectorStore.`);
    console.log("\nVector store setup complete!");
    
    // Example query
    const question = "What landmark was constructed in the 1800s?";
    console.log(`\nQuery: ${question}`);
    
    // Use the vector store to retrieve relevant documents
    const retriever = vectorStore.asRetriever({
      k: 3, // Number of documents to retrieve
    });
    const retrievedDocs = await retriever.getRelevantDocuments(question);
    
    // Print the retrieved documents
    console.log(`\nRetrieved ${retrievedDocs.length} relevant documents:`);
    retrievedDocs.forEach((doc, i) => {
      console.log(`${i + 1}. ${doc.pageContent}`);
    });
    
    // Extract just the text content from the retrieved documents
    const retrievedTexts = retrievedDocs.map(doc => doc.pageContent);
    const context = retrievedTexts.join('\n\n');
    
    // Use the same TeapotAI instance to answer the question with context
    const answer = await teapot.query(question, context);
    console.log(`\nAnswer: ${answer}`);
    
    // Another example query
    const question2 = "What is the tallest building mentioned in our documents?";
    console.log(`\n\nQuery: ${question2}`);
    
    // Retrieve documents for the second question
    const retrievedDocs2 = await retriever.getRelevantDocuments(question2);
    
    // Print the retrieved documents for the second question
    console.log(`\nRetrieved ${retrievedDocs2.length} relevant documents:`);
    retrievedDocs2.forEach((doc, i) => {
      console.log(`${i + 1}. ${doc.pageContent}`);
    });
    
    // Extract just the text content from the retrieved documents
    const retrievedTexts2 = retrievedDocs2.map(doc => doc.pageContent);
    const context2 = retrievedTexts2.join('\n\n');
    
    // Use the same TeapotAI instance to answer the second question with context
    const answer2 = await teapot.query(question2, context2);
    console.log(`\nAnswer: ${answer2}`);
    
  } catch (error) {
    console.error("Error:", error);
  }
}

main();