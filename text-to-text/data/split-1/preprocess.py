import json
import sys

from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP("stanford-corenlp-full-2018-10-05")

def preprocess(squad_file, mpqg_file):
    with open(squad_file) as f:
        squad_dataset = json.load(f, encoding='utf-8')
    squad_dataset_dict = {}
    paragraphs = []
    index = 0
    for topic in squad_dataset:
        examples = topic["paragraphs"]
        for example in examples:
            for qas in example['qas']:
                assert qas['id'] not in squad_dataset_dict
                squad_dataset_dict[qas['id']] = {"answers": qas['answers'], "question": qas['question'], "paragraph_id": index}

            paragraphs.append(example['context'])
            index += 1
                

    with open(mpqg_file) as f:
        mpqg_dataset = json.load(f)

    answer_spans_multiple_sentences = 0
    counter = 0
    for index, mpqg_example in enumerate(mpqg_dataset):
        example_id = mpqg_example['id']
        squad_example = squad_dataset_dict[example_id]

        paragraph = paragraphs[squad_example['paragraph_id']].encode('utf-8')
        offset = len(paragraph) - len(paragraph.lstrip())

        ans_start, ans_end = None, None
        for ans in squad_example['answers']:
            if ans['text'] == mpqg_example['text3']:
                ans_start = ans['answer_start']
                ans_end = ans['answer_start'] + len(ans['text'])

        assert ans_start != None
        assert ans_end != None

        output = nlp.annotate(paragraph, properties={'annotators': 'tokenize,ssplit,ner,pos', 'outputFormat': 'json'})
        if type(output) is str or type(output) is unicode:
            output = json.loads(output, strict=False)

        IO_parag = []
        IO_sent = []
        POS_parag = []
        POS_sent = []
        NER_parag = []
        NER_sent = []
        sentence_toks = []
        paragraph_toks = []
        first_sent = None
        last_sent = None
        for sent_num, sent in enumerate(output['sentences']):
            sent_start = sent['tokens'][0]['characterOffsetBegin'] + offset
            sent_end = sent['tokens'][-1]['characterOffsetEnd'] + offset
            if ans_start >= sent_start and ans_start < sent_end:
                first_sent = sent_num
            if ans_end > sent_start and ans_end <= sent_end:
                last_sent = sent_num
            if first_sent != None and last_sent != None:
                for i in range(first_sent, last_sent+1):
                    for tok in output['sentences'][i]['tokens']:
                        sentence_toks.append(tok['word'].lower())
                        POS_sent.append(tok['pos'])
                        NER_sent.append(tok['ner'])
                if first_sent != last_sent:
                    answer_spans_multiple_sentences += 1
                break

        for sent_num, sent in enumerate(output['sentences']):
            sent = sent['tokens']
            if sent_num <= last_sent and sent_num >= first_sent:
                for tok in sent:
                    paragraph_toks.append(tok['word'].lower())
                    POS_parag.append(tok['pos'])
                    NER_parag.append(tok['ner'])
                    if (tok['characterOffsetBegin'] + offset >= ans_start and tok['characterOffsetEnd'] + offset <= ans_end) or \
                       (tok['characterOffsetBegin'] + offset <= ans_start and tok['characterOffsetEnd'] + offset > ans_start) or \
                       (tok['characterOffsetBegin'] + offset < ans_end and tok['characterOffsetEnd'] + offset > ans_end):
                        IO_sent.append(1)
                        IO_parag.append(1)
                    else:
                        IO_sent.append(0)
                        IO_parag.append(0)
            else:
                for tok in sent:
                    paragraph_toks.append(tok['word'].lower())
                    POS_parag.append(tok['pos'])
                    NER_parag.append(tok['ner'])
                    IO_parag.append(0)
        
        assert len(sentence_toks) == len(IO_sent)
        assert len(sentence_toks) == len(POS_sent)
        assert len(sentence_toks) == len(NER_sent)
        assert len(paragraph_toks) == len(IO_parag)
        assert len(paragraph_toks) == len(POS_parag)
        assert len(paragraph_toks) == len(NER_parag)
        assert sentence_toks[IO_sent.index(1):IO_sent.index(1)+sum(IO_sent)] == paragraph_toks[IO_parag.index(1):IO_parag.index(1)+sum(IO_parag)]

        if sentence_toks[IO_sent.index(1):IO_sent.index(1)+sum(IO_sent)] != mpqg_example['annotation3']['toks'].lower().split():
            counter += 1
            print(counter)
            print("Answer from IO feature: ", " ".join(sentence_toks[IO_sent.index(1):IO_sent.index(1)+sum(IO_sent)]))
            print("Answer from MPQG: ", mpqg_example['annotation3']['toks'].lower())
            print('--------------------------------------------------------------')
               
        mpqg_example['text1_parag'] = paragraph
        mpqg_example['IO_sent'] = IO_sent
        mpqg_example['IO_parag'] = IO_parag
        mpqg_example['annotation1'] = {}
        mpqg_example['annotation1']['toks_sent'] = " ".join(sentence_toks)
        mpqg_example['annotation1']['toks_parag'] = " ".join(paragraph_toks)
        mpqg_example['annotation1']['NER_sent'] = " ".join(NER_sent)
        mpqg_example['annotation1']['NER_parag'] = " ".join(NER_parag)
        mpqg_example['annotation1']['POS_sent'] = " ".join(POS_sent)
        mpqg_example['annotation1']['POS_parag'] = " ".join(POS_parag)
        

        if index % 500 == 0:
            print(index)
            sys.stdout.flush()
    
    print('Total # mismatch: ', counter)
    print('Ratio of examples with answers that span multiple sentences:', answer_spans_multiple_sentences/index)

    with open('processed_' + squad_file, 'w') as outfile:
        json.dump(mpqg_dataset, outfile)


def main():
    preprocess('train.json', 'train_sent_pre.json')
    preprocess('dev.json', 'dev_sent_pre.json')
    preprocess('test.json', 'test_sent_pre.json')


if __name__ == '__main__':
    main()

