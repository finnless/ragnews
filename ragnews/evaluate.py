'''
For evaluating the quality of the RAG pipeline using the
Hairy Trumpet dataset.
'''

import ragnews
import re


class RAGClassifier:
    def __init__(self):
        '''
        Initialize the RAGClassifier. The __init__ function should take
        an input that specifies the valid labels to predict.
        The class should be a "predictor" following the scikit-learn interface.
        You will have to use the ragnews.rag function internally to perform the prediction.
        '''
        pass

    def predict(self, masked_text: str):
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
        rag_result = ragnews.rag(rag_query, db)
        system = (
            'You are a helpful assistant that predicts the answers of the masked text '
            'based only on the context provided. '
            'Masked text is in the format of [MASK0], [MASK1], etc. '
            'Think through your answer step-by-step in no more than 50 words. '
            'Provide the final answer for each masked text with no leading text on a new line at the end of your response.\n\n'
        )
        user = (
            '# Context\n'
            '{context}\n\n'
            '# Masked Text\n'
            '{masked_text}'
        )
        user = user.format(context=rag_result, masked_text=masked_text)
        print('user:\n', user)
        output = ragnews.run_llm(system, user)

        return output