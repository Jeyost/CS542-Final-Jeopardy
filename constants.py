DATAPATH_MAIN           = 'jeopardy_clue_dataset-master'
DATAPATH_SEASONS        = f'{DATAPATH_MAIN}/seasons'
LUCENE_INDEX            = '/s/bach/b/class/cs542/cs542a/indexes/lucene-index.enwiki-20180701-paragraphs'
NUM_CONTEXTS            = 2
DATA_ROUND              = 0
DATA_CLUE_VALUE         = 1
DATA_DAILY_DOUBLE_VALUE = 2
DATA_CATEGORY           = 3
DATA_COMMENTS           = 4
DATA_ANSWER             = 5
DATA_QUESTION           = 6
DATA_AIR_DATE           = 7
DATA_NOTES              = 8

CSV_HEADER              = ["SEASON", "CLUE VALUE","CLUE CATEGORY","QUESTION","OUR MODEL_ANSWER","REAL ANSWER","CORRECT","CONTEXT USED"]
OUR_DATA_SEASON         = 0
OUR_DATA_CLUE_VALUE     = 1
OUR_DATA_CATEGORY       = 2
OUR_DATA_QUESTION       = 3 # the answer in jep, ie the thing the person is given
OUR_DATA_ANSWER         = 4 # the thing the contestant responds with
CSV_FILE_NO_TRAINING    =  'no_training_all_question_answers.csv'