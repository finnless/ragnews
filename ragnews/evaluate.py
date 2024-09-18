'''
For evaluating the quality of the RAG pipeline using the
Hairy Trumpet dataset.
'''

import ragnews
import re
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
# TODO add logging to everything

class RAGClassifier:
    def __init__(self):
        '''
        Initialize the RAGClassifier. The __init__ function should take
        an input that specifies the valid labels to predict.
        The class should be a "predictor" following the scikit-learn interface.
        You will have to use the ragnews.rag function internally to perform the prediction.
        '''
        pass

    def predict(self, masked_text: str, attempt=0):
        '''
        Predict the labels of the documents.
        >>> model = RAGClassifier()
        >>> model.predict('There is no mask token')
        []
        >>> model.predict('[MASK0] is the democratic nominee for president in 2024')
        ['Harris']
        >>> model.predict('[MASK0] is the democratic nominee and [MASK1] is the republican nominee')
        ['Harris', 'Trump']
        '''
        if not re.search(r'\[MASK\d+\]', masked_text):
            return []
        db = ragnews.ArticleDB('ragnews.db')
        rag_query = re.sub(r'\[MASK\d+\]', '', masked_text).strip()
        # TODO fix this up
        masks = list(set(re.findall(r'\[MASK\d+\]', masked_text)))
        rag_query = rag_query + '\n\nI am doing a Cloze test on the above text. Fill in the blanks. Who are: ' + ', '.join(masks)
        rag_result = ragnews.rag(rag_query, db)
        system = (
            'You are a helpful assistant that predicts the answers of the masked text '
            'based only on the context provided. '
            'Masked text is in the format of [MASK0], [MASK1], etc. '
            'Think through your answer step-by-step in no more than 50 words. '
            'If your answer is a person, provide their last name ONLY. '
            'Once you have a final answer for each mask, provide each answer on a new line at the end of your response after a a divider line (---).\n'
            'Example:\n'
            '...\n'
            'Therefore [MASK0] is Alice and [MASK1] is Bob.\n\n'
            '---\n'
            'Alice\n'
            'Bob'
        )
        user = (
            '# Context\n'
            '{context}\n\n'
            '# Masked Text\n'
            '{masked_text}'
        )
        user = user.format(context=rag_result, masked_text=masked_text)
        logging.debug('user:\n%s', user)
        output = ragnews.run_llm(system, user)
        logging.debug('output:\n%s', output)

        # if the output is not in the correct format, try again
        if '---' not in output and attempt < 3:
            logging.warning('error parsing output, trying again... attempt: %d', attempt)
            return self.predict(masked_text, attempt=attempt+1)
        elif '---' not in output and attempt >= 3:
            return []
        
        # output parser
        output_lines = output.strip().split('---')
        results = [line for line in output_lines[-1].split('\n') if line.strip()]

        return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    args = parser.parse_args()

    with open(args.data, 'r') as f:
        data = [json.loads(line) for line in f]

    model = RAGClassifier()

    success = 0
    failure = 0
    # TODO remove limit
    data = data[:2]
    for d in data:
        prediction = model.predict(d['masked_text'])
        print('predicted labels:', prediction)
        print('actual labels:', d['masks'])
        print('-' * 100)
        print()
        if len(prediction) == len(d['masks']):
            if all(
                mask.lower() in pred.lower() 
                for mask, pred in zip(d['masks'], prediction)
            ):
                success += 1
            else:
                failure += 1
        else:
            failure += 1
    
    logging.debug('success: %d', success)
    logging.debug('failure: %d', failure)


# for the querying, could send rewriter the question who is [MASK0] at the end

'''
When the file is run as a script, it should take a command line argument which is the path to a HairyTrumpet data file.

Extract all the possible labels from the datafile, then build a RAGClassifier over these labels.
Run the predict method on each instance inside the file, and compute the accuracy of the predictions.


Example hairytrumpet datafile:

```jsonl
{"masked_text": "Three Democratic-held seats up for election are in the heavily Republican-leaning states of Montana, Ohio, and West Virginia, all of which were won comfortably by [MASK0] in both 2016 and 2020.", "masks": ["Trump"]}
{"masked_text": "On July 13, 2024, during a campaign rally in Butler, Pennsylvania, presidential candidate [MASK0] was shot at in a failed assassination attempt.", "masks": ["Trump"]}
```
'''