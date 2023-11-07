from bertserini.bertserini.reader.base import Question, Context
from bertserini.bertserini.reader.bert_reader import BERT
from bertserini.bertserini.utils.utils_new import get_best_answer

model_name = "rsvp-ai/bertserini-bert-base-squad"
tokenizer_name = "rsvp-ai/bertserini-bert-base-squad"
bert_reader = BERT(model_name, tokenizer_name)

# Here is our question:
question = Question("Why did Mark Twain call the 19th century the glied age?")

# Option 1: fetch some contexts from Wikipedia with Pyserini
from bertserini.bertserini.retriever.pyserini_retriever import retriever, build_searcher
searcher = build_searcher("indexes/lucene-index.enwiki-20180701-paragraphs")
contexts = retriever(question, searcher, 10)

# Option 2: hard-coded contexts
contexts = [Context('The "Gilded Age" was a term that Mark Twain used to describe the period of the late 19th century when there had been a dramatic expansion of American wealth and prosperity.')]

# Either option, we can ten get the answer candidates by reader
# and then select out the best answer based on the linear 
# combination of context score and phase score
candidates = bert_reader.predict(question, contexts)
answer = get_best_answer(candidates, 0.45)
print(answer.text)