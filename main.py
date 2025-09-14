from transformers import pipeline

pipe = pipeline("text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

while True:
    user_input = input("Enter your text: ")
    if user_input.lower() == 'quit':
        break
    result = pipe(user_input)
    print("Result:", result)