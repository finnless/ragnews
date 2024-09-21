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
    def __init__(self, valid_labels):
        '''
        Initialize the RAGClassifier. The __init__ function should take
        an input that specifies the valid labels to predict.
        The class should be a "predictor" following the scikit-learn interface.
        You will have to use the ragnews.rag function internally to perform the prediction.
        '''
        self.valid_labels = valid_labels

    def _extract_cloze_keywords(text, seed=0, temperature=None):
        r'''
        This is a helper function for RAG with the Cloze test.
        Given an input text,
        this function extracts the keywords that will be used to perform the search for articles that will be used in RAG.

        >>> text1 = """On July 13, 2024, during a campaign rally in Butler, Pennsylvania, presidential candidate [MASK0] was shot at in a failed assassination attempt. The gunfire caused minor damage to [MASK0]'s upper right ear, while one spectator was killed and two others were critically injured. On September 15, 2024, the security detail of [MASK0] spotted an armed man while the former president was touring his golf course in West Palm Beach. They opened fire on the suspect, which fled on a vehicle and was later captured thanks to the contribution of an eyewitness. In the location where the suspect was spotted, the police retrieved an AK-47-style rifle with a scope, two rucksacks and a GoPro."""
        >>> RAGClassifier._extract_cloze_keywords(text1)
        'campaign, shooting, Pennsylvania, presidential, assassination'
        >>> text2 = """After a survey by the Associated Press of Democratic delegates on July 22, 2024, [MASK0] became the new presumptive candidate for the Democratic party, a day after declaring her candidacy. She would become the official nominee on August 5 following a virtual roll call of delegates."""
        >>> RAGClassifier._extract_cloze_keywords(text2)
        '"2024 democratic presumptive candidate"'
        '''
        # TODO put OR between simmilar keywords
        # TODO use the context from the predicate of the relevant sentence to create keywords most likely to return articles that mention the subject
        system = 'You are helping a team do Cloze test. Your task is to write a search query that will retrieve relevant articles from a news database. Craft your query with only the most important keywords related to the masked text. Limit to 5 keywords at most. Return only search keywords, do not include any other text or numbers.'
        return ragnews.run_llm(system, text, seed=seed, temperature=temperature)

    def predict(self, masked_text: str, attempt=0):
        '''
        Predict the labels of the documents.
        >>> model = RAGClassifier(['Trump', 'Biden', 'Harris'])
        >>> model.predict('There is no mask token')
        []
        >>> model.predict('[MASK0] is the democratic nominee for president in 2024')
        ['Harris']
        >>> model.predict('[MASK0] is the democratic nominee and [MASK1] is the republican nominee')
        ['Harris', 'Trump']
        >>> text1 = """On July 13, 2024, during a campaign rally in Butler, Pennsylvania, presidential candidate [MASK0] was shot at in a failed assassination attempt. The gunfire caused minor damage to [MASK0]'s upper right ear, while one spectator was killed and two others were critically injured. On September 15, 2024, the security detail of [MASK0] spotted an armed man while the former president was touring his golf course in West Palm Beach. They opened fire on the suspect, which fled on a vehicle and was later captured thanks to the contribution of an eyewitness. In the location where the suspect was spotted, the police retrieved an AK-47-style rifle with a scope, two rucksacks and a GoPro."""
        >>> model.predict(text1)
        ['Trump']
        >>> text2 = """After a survey by the Associated Press of Democratic delegates on July 22, 2024, [MASK0] became the new presumptive candidate for the Democratic party, a day after declaring her candidacy. She would become the official nominee on August 5 following a virtual roll call of delegates."""
        >>> model.predict(text2)
        ['Harris']
        '''
        db = ragnews.ArticleDB('ragnews.db')
        
        if not re.search(r'\[MASK\d+\]', masked_text):
            return []
        masks = list(set(re.findall(r'\[MASK\d+\]', masked_text)))
        # Set dynamic example names depending on the number of masks
        example_names = ['Alice', 'Bob', 'Eve', 'Mallory', 'Trent']
        example_mapping = ', '.join([f'[MASK{i}] is {example_names[i]}' for i in range(len(masks))])
        example_answers = '\n'.join(example_names[:len(masks)])
        
        system = (
            'You are a helpful assistant that predicts the answers of the masked text '
            'based only on the context provided. '
            'Masked text is in the format of {masks}. '
            'The answers to choose from are: {valid_labels}. '
            'Think through your answer step-by-step in no more than 50 words. '
            'If your answer is a person, provide their last name ONLY. '
            'As soon as you have a final answer for all masks, provide each answer on a new line at the end of your response inside a single <answer> tag like this:\n'
            '...(your reasoning here)...\n'
            'Therefore {example_mapping}.\n\n'
            '<answer>\n'
            '{example_answers}\n'
            '</answer>'
        )
        system = system.format(masks=' '.join(masks),
                               example_mapping=example_mapping,
                               example_answers=example_answers,
                               valid_labels=self.valid_labels,
                               )
        keywords = RAGClassifier._extract_cloze_keywords(masked_text)
        # TODO make temperature and other hyperparameters tunable
        output = ragnews.rag(masked_text, db, keywords=keywords, system=system, temperature=0.5, stop='</answer>')
        # if the output is not in the correct format, try again
        if '<answer>' not in output and attempt < 3:
            logging.warning('error parsing output, trying again... attempt: %d', attempt)
            return self.predict(masked_text, attempt=attempt+1)
        elif '<answer>' not in output and attempt >= 3:
            return []
        
        # output parser
        output_lines = output.strip().split('<answer>')
        # TODO deal with </answer>
        results = [line for line in output_lines[-1].split('\n') if line.strip()]

        return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    args = parser.parse_args()

    with open(args.data, 'r') as f:
        data = [json.loads(line) for line in f]

    # TODO make valid_labels dynamic. https://github.com/mikeizbicki/cmc-csci181-languages/issues/19
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