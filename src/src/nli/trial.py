from transformers import pipeline
pipe = pipeline("text-classification",model="tasksource/ModernBERT-base-nli")


print(pipe([dict(text="Who is a binge drinker?\n\nAnswer Choices:\nA. The old-timer\nB. Cannot be determined\nC. The twenty-something\n\nAlthough both individuals mentioned beer as a way to relax, there's no explicit information about how much they consume or the frequency of binge drinking.",
  text_pair='The answer is B: Cannot be determined')]) #list of (premise,hypothesis)
)

