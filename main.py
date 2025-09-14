from transformers import pipeline
'''
pipe = pipeline("text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

while True:
    user_input = input("Enter your text: ")
    if user_input.lower() == 'quit':
        break
    result = pipe(user_input)
    print("Result:", result)
 '''

pipe = pipeline("image-classification", model="google/vit-base-patch16-224")
image_path = input("Enter image filename: ")
result = pipe(image_path)

best_guess = result[0]
label = best_guess['label']
confidence = best_guess['score'] * 100

clean_output = f"{label} ({confidence:.1f}% confidence)"
print(clean_output)