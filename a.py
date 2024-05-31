import whisper
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
!pip install openai-whisper transformers torch datasets faiss-cpu
model = whisper.load_model("medium")

def transcribe_audio(audio_path):
    result = model.transcribe(audio_path)
    return result["text"]
  
tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", index_name="exact")
rag_model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq")

def generate_response(question):
    inputs = tokenizer(question, return_tensors="pt")
    generated = rag_model.generate(**inputs)
    response = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    return response
  
def handle_audio_query(audio_path):
    question = transcribe_audio(audio_path)
    print(f"Transcribed Text: {question}")

    response = generate_response(question)
    return response

audio_path = "my_file.wav"
response = handle_audio_query(audio_path)
print(f"Response: {response}")

def translate_and_summarize(text):
    inputs = tokenizer(text, return_tensors="pt")
    generated = rag_model.generate(**inputs)
    translation_summary = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    return translation_summary
  
text_to_translate_summarize = "Your text here"  # Replace with the text to translate and summarize
translated_summarized_text = translate_and_summarize(text_to_translate_summarize)
print(f"Translated and Summarized Text: {translated_summarized_text}")
