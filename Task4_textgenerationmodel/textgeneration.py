from transformers import pipeline, set_seed

# Initialize the text generation pipeline with GPT-2
generator = pipeline('text-generation', model='gpt2')

# User prompt (you can change this to test different topics)
prompt = input("Enter your topic or sentence prompt: ")
# Generate text
output = generator(prompt, max_length=150, num_return_sequences=1)

print("\nGenerated Text:\n")
print(output[0]['generated_text'])
