**Supervised NER technqiue**
Several transformer based approaches for NER:
1)Vanilla transformer
2)BertForTokenClassification,Roberta,distillbert,deberta
https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BERT/Custom_Named_Entity_Recognition_with_BERT.ipynb
3)


**Unsupervised/Zero-Few shot learning NER technqiue**
## 1) Gliner
## 2) LLM model & OpenAI (models GPT3.5,GOT4) with prompt : Few shot learning with encoder based models. 
## 3) Spacy llms 
4) llm ner
4) Langchains (LLMChain) with prompt template i.e. create_extraction_chain - 
5) Langchain Chatmodels with systemprompttemplate,aipromptemplate,humanprompttemplate
https://python.langchain.com/v0.1/docs/use_cases/extraction/
https://www.youtube.com/watch?v=OagbDJvywJI
https://medium.com/@grisanti.isidoro/named-entity-recognition-with-llms-extract-conversation-metadata-94d5536178f2
llmner - chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://arxiv.org/pdf/2406.04528v1



Tokens --> POS tags --> Chunking --> Named Entity Detection and Recognition --> Relation Extraction --> Understanding and Evaluating Relationships

Chunking:-
https://medium.com/@dipansutech/what-is-chunking-in-nlp-dipansu-tech-506f518e9e6a
https://towardsdatascience.com/chunking-in-nlp-decoded-b4a71b2b4e24
https://www.quora.com/What-is-the-difference-between-tagger-chunker-and-NER


NER Implementation

1)Unsupervised/few shot learning/Zero shot learning for generation of labelling data set i.e. IOB notation

2)Supervised fine tuning on labelled NER data.

3) Metric to evaluate NER



Relationship extraction 
1)Knowledge graphs - LLMGraphTransformer ( llama3 with knowledge graphs ) , refer bookmarks
2)Langchains i.e. LLMchain with relationships as prompt
