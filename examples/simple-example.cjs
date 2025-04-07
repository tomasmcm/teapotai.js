// Simple TeapotAI example for Node.js
const { TeapotAI } = require('../../dist/teapotai.cjs');

async function main() {
  try {
    console.log("Initializing TeapotAI...");
    
    // Initialize with default settings
    const teapot = await TeapotAI.fromPretrained({
      settings: {
        verbose: true
      }
    });
    
    // Simple query example
    const question = "What is the capital of France?";
    console.log(`\nQuery: ${question}`);
    
    const answer = await teapot.query(question);
    console.log(`Answer: ${answer}`);
    
    // Chat conversation example
    console.log("\nStarting chat conversation...");
    const conversation = [
      { role: "system", content: "You are a helpful assistant." },
      { role: "user", content: "Hello! Who are you?" },
      { role: "assistant", content: "I am Teapot" },
      { role: "user", content: "What can you help me with?" }
    ];
    
    const chatResponse = await teapot.chat(conversation);
    console.log(`Chat response: ${chatResponse}`);
    
  } catch (error) {
    console.error("Error:", error);
  }
}

main();