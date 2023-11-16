import csv
import logging
from constants import *
from discord_logging.handler import DiscordHandler
import os
from bertserini.reader.base import Question, Context
from bertserini.reader.bert_reader import BERT
from bertserini.utils.utils_new import get_best_answer
from bertserini.experiments.args import *
from bertserini.retriever.pyserini_retriever import retriever, build_searcher

def main(logger):
    num_rows = 0 
    data = load_data()
    file_exists = os.path.exists(CSV_FILE_NO_TRAINING)
    if file_exists:
        with open(CSV_FILE_NO_TRAINING, 'r', newline='') as csvfile:
            num_rows = len(list(csv.reader(csvfile))) -1 # get ride of header
    mode_of_writting = 'a' if file_exists else 'w'
    with open(CSV_FILE_NO_TRAINING, mode_of_writting) as csvfile:
        csv_writer = csv.writer(csvfile)
        if not file_exists:
            csv_writer.writerow(CSV_HEADER)
        answer_questions(data[num_rows:], logger,csvfile,csv_writer)

    

def answer_questions(questions,logger,csv_file, csv_writer):
    args.model_name_or_path = "rsvp-ai/bertserini-bert-base-squad"
    args.tokenizer_name = "rsvp-ai/bertserini-bert-base-squad"
    bert_reader = BERT(args)
    args.index_path = LUCENE_INDEX
    searcher = build_searcher(args)    
    total_num_questions = len(questions) 
    for i,  question in enumerate(questions):
        if i %1000 == 0: logger.info(f"Working on {i}/{total_num_questions}: percent done: {i/total_num_questions}%")
        answer = answer_a_question(Question(question[OUR_DATA_QUESTION]), bert_reader, searcher)
        write_question_answer_to_csv(question,answer,csv_file,csv_writer)
        break


def answer_a_question(question, bert_reader, searcher):
    contexts = retriever(question, searcher, NUM_CONTEXTS)
    candidates = bert_reader.predict(question, contexts)
    return get_best_answer(candidates, 0.45)

def write_question_answer_to_csv(question, answer, csv_file, cvs_writer):
    correct = answer.text in question[OUR_DATA_ANSWER]
    row = [question[OUR_DATA_SEASON],question[OUR_DATA_CLUE_VALUE],question[OUR_DATA_CATEGORY],question[OUR_DATA_QUESTION], 
           None if answer.text =="" else answer.text,question[OUR_DATA_ANSWER], correct, answer.metadata['context']]
    cvs_writer.writerow(row)
    csv_file.flush()


def load_data():
   questions = []
   for filename in os.listdir(DATAPATH_SEASONS):
        file_path = os.path.join(DATAPATH_SEASONS, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as season:
                season_number = filename[6:-4]
                print(season_number)
                for line in season.readlines()[1:]:
                    question =  line.split('\t')
                    questions.append([season_number, question[DATA_CLUE_VALUE], question[DATA_CATEGORY], question[DATA_QUESTION], question[DATA_ANSWER]])
   return questions


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    webhook_url = os.environ["DISCORD_RESEARCH_WEBHOOK_URL"]
    discord_format = logging.Formatter("%(message)s")
    discord_handler = DiscordHandler('Adama', webhook_url=webhook_url)
    discord_handler.setFormatter(discord_format)
    logger.addHandler(discord_handler)
    try:
        main(logger)
        logger.info("FINISHED QUESTIONS")
    except Exception as e: 
        logger.error('ERROR OCCURED')
        logger.error(e)
        logger.error("SHUTTING DOWN")

