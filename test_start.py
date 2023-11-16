from bertserini.reader.base import Question, Context
from bertserini.reader.bert_reader import BERT
from bertserini.utils.utils_new import get_best_answer
from bertserini.experiments.args import *
from constants import *
args.model_name_or_path = "rsvp-ai/bertserini-bert-base-squad"
args.tokenizer_name = "rsvp-ai/bertserini-bert-base-squad"
bert_reader = BERT(args)

# Here is our question:
question = Question("What is the meaning of life?")

# Option 1: fetch some contexts from Wikipedia with Pyserini
from bertserini.retriever.pyserini_retriever import retriever, build_searcher
args.index_path = LUCENE_INDEX
searcher = build_searcher(args)
contexts = retriever(question, searcher, NUM_CONTEXTS)

# Option 2: hard-coded contexts
#contexts = [Context('The "Gilded Age" was a term that Mark Twain used to describe the period of the late 19th century when there had been a dramatic expansion of American wealth and prosperity.')]

# Either option, we can ten get the answer candidates by reader
# and then select out the best answer based on the linear 
# combination of context score and phase score
candidates = bert_reader.predict(question, contexts)
answer = get_best_answer(candidates, 0.45)
print(answer.text)
