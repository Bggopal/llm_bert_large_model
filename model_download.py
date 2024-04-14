from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# Specify the model identifier
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"

# Download the model and tokenizer
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save the model and tokenizer to a local directory
model.save_pretrained("/path/to/save/model")
tokenizer.save_pretrained("/path/to/save/tokenizer")
