import { describe, it, expect, beforeEach } from 'vitest';
import { TeapotAI } from '../src/teapotai';

describe('TeapotAI', () => {
  let teapot;

  beforeEach(async () => {
    teapot = await TeapotAI.fromPretrained({
      settings: {
        useRag: false,
        verbose: true
      }
    });
  }, 0);

  it('should get Eiffel Tower height with context', async () => {
    const context = `
      The Eiffel Tower is a wrought iron lattice tower in Paris, France. It was designed by Gustave Eiffel and completed in 1889.
      It stands at a height of 330 meters and is one of the most recognizable structures in the world.
    `;
    const response = await teapot.query("What is the height of the Eiffel Tower?", context);
    expect(response).toContain("Eiffel Tower");
    expect(response).toContain("330 meters");
  }, 0);

  it('should handle missing height information', async () => {
    const context = `
      The Eiffel Tower is a wrought iron lattice tower in Paris, France. It was designed by Gustave Eiffel and completed in 1889.
    `;
    const response = await teapot.query("What is the weather in Paris?", context);
    expect(response).toBe("I apologize, but I'm not capable of providing weather data.");
  }, 0);

  it('should use RAG for landmark questions', async () => {
    const documents = [
      "The Eiffel Tower is located in Paris, France. It was built in 1889 and stands 330 meters tall.",
      "The Amazon Rainforest is the largest tropical rainforest in the world, covering over 5.5 million square kilometers.",
      "The Grand Canyon is a natural landmark located in Arizona, USA, carved by the Colorado River.",
      "Mount Everest is the tallest mountain on Earth, located in the Himalayas along the border between Nepal and China.",
      "The Colosseum in Rome, Italy, is an ancient amphitheater known for its gladiator battles.",
      "The Sahara Desert is the largest hot desert in the world, located in North Africa.",
      "The Nile River is the longest river in the world, flowing through northeastern Africa.",
      "The Empire State Building is an iconic skyscraper in New York City that was completed in 1931 and stands at 1454 feet tall."
    ];

    const teapotWithDocs = await TeapotAI.fromPretrained({
      documents,
      settings: { useRag: true, verbose: true }
    });
    const response = await teapotWithDocs.chat([
      { role: "system", content: "You are an agent designed to answer facts about famous landmarks." },
      { role: "user", content: "What landmark was constructed in the 1800s?" }
    ]);

    expect(response).toBe("The Eiffel Tower was constructed in 1889.");
  }, 0);

  it('should extract apartment information', async () => {
    const apartmentDescription = `
      This spacious 2-bedroom apartment is available for rent in downtown New York. The monthly rent is $2500.
      It includes 1 bathrooms and a fully equipped kitchen with modern appliances.

      Pets are welcome!

      Please reach out to us at 555-123-4567 or john@realty.com
    `;

    const schema = {
      rent: { type: 'number', description: 'the monthly rent in dollars' },
      bedrooms: { type: 'number', description: 'the number of bedrooms' },
      bathrooms: { type: 'number', description: 'the number of bathrooms' },
      phone_number: { type: 'string' }
    };

    const extractedInfo = await teapot.extract(schema, "", apartmentDescription);
    console.log(extractedInfo)

    expect(extractedInfo.rent).toBe(2500);
    expect(extractedInfo.bedrooms).toBe(2);
    expect(extractedInfo.bathrooms).toBe(1);
    expect(extractedInfo.phone_number).toBe('555-123-4567');
  }, 0);
});