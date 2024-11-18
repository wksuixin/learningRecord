from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import AutoModelForSequenceClassification

'''
pipeline三个步骤：
1. 预处理，如分词
2. 输入模型
3. 后处理
'''
classifier = pipeline("sentiment-analysis")
result = classifier("I'm writting a code of hugging face!")
print(result)

generator = pipeline("text-generation")
result = generator("I just saw a beautiful girl!", num_return_sequences=2, max_length=100)
print(result)

'''
分步骤测试
'''
checkpoint_path = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
raw_inputs = ["I'm writting a code of hugging face!", "I love this so much"]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)

model = AutoModel.from_pretrained(checkpoint_path)
outputs = model(**inputs)
print(outputs)

classify_model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
outputs = classify_model(**inputs)
print(outputs)
