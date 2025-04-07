// RAG (Retrieval Augmented Generation) example with TeapotAI for Node.js
const { TeapotAI } = require('../../dist/teapotai.cjs');

async function main() {
  try {
    console.log("Initializing TeapotAI with RAG...");
    
    // Example documents for knowledge retrieval
    const documents = [
      "The Eiffel Tower is located in Paris, France. It was built in 1889 and stands 330 meters tall.",
      "The Great Wall of China is a historic fortification that stretches over 13,000 miles.",
      "The Empire State Building is an iconic skyscraper in New York City that was completed in 1931 and stands at 1,454 feet tall.",
      "The Sahara Desert is the largest hot desert in the world, located in North Africa.",
      "The Nile River is the longest river in the world, flowing through northeastern Africa.",
      "The Amazon Rainforest is the largest tropical rainforest in the world, covering over 5.5 million square kilometers."
    ];
    
    // Initialize with documents and RAG settings
    const teapot = await TeapotAI.fromPretrained({
      documents,
      settings: {
        useRag: true,
        ragNumResults: 3,
        ragSimilarityThreshold: 0.5,
        verbose: true
      }
    });
    
    console.log("\nDocuments loaded and embedded successfully!");
    
    // Example query using RAG
    const question = "What landmark was constructed in the 1800s?";
    console.log(`\nQuery: ${question}`);
    
    const answer = await teapot.query(question);
    console.log(`Answer: ${answer}`);
    
    // Another example
    const question2 = "What is the tallest building mentioned in our documents?";
    console.log(`\nQuery: ${question2}`);
    
    const answer2 = await teapot.query(question2);
    console.log(`Answer: ${answer2}`);
    
  } catch (error) {
    console.error("Error:", error);
  }
}

main();