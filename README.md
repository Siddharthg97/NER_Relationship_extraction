## NER Implementation

1)Unsupervised/few shot learning/Zero shot learning for generation of labelling data set i.e. IOB notation   <br />
2)Supervised fine tuning on labelled NER data     <br />
3) Metric to evaluate NER  <br/>


### Unsupervised/Zero-Few shot learning NER technqiue
1) **Gliner**
   https://github.com/urchade/GLiNER?tab=readme-ov-file

2) **LLM model & OpenAI (models GPT3.5,GOT4) with prompt : Few shot learning with decoder based models from hugging face**
3) **Spacy llms** <br />
4) **llm ner** - llmner - chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://arxiv.org/pdf/2406.04528v1 <br />
5) **Langchains (LLMChain) with prompt template** i.e. create_extraction_chain   <br />
6) **Langchain Chatmodels with systemprompttemplate,aipromptemplate,humanprompttemplate**  <br />
https://python.langchain.com/v0.1/docs/use_cases/extraction/
https://www.youtube.com/watch?v=OagbDJvywJI
https://medium.com/@grisanti.isidoro/named-entity-recognition-with-llms-extract-conversation-metadata-94d5536178f2

### Supervised NER technqiue
Several transformer based approaches for NER: <br />
1)Vanilla transformer  <br />
2)BertForTokenClassification,Roberta,distillbert,deberta <br />
https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BERT/Custom_Named_Entity_Recognition_with_BERT.ipynb <br />
3)

**Important Note** <br />
If data size is larger than token limit then chunking of data needs to be done for extraction of entities.


Tokens --> POS tags --> Chunking --> Named Entity Detection and Recognition --> Relation Extraction --> Understanding and Evaluating Relationships


Chunking:-

https://medium.com/@dipansutech/what-is-chunking-in-nlp-dipansu-tech-506f518e9e6a
https://towardsdatascience.com/chunking-in-nlp-decoded-b4a71b2b4e24
https://www.quora.com/What-is-the-difference-between-tagger-chunker-and-NER


***Relationship extraction***

**1)Knowledge graphs - LLMGraphTransformer ( llama3 with knowledge graphs ) , refer bookmarks** <br />
Enhancing LLMs Inference with Knowledge Graphs - https://medium.com/@bijit211987/enhancing-llms-inference-with-knowledge-graphs-7140b3c3d683   <br /> 
https://www.geeksforgeeks.org/relationship-extraction-in-nlp/  <br />
https://neo4j.com/labs/genai-ecosystem/llm-graph-builder/ <br />
**2)Buidling knowledge graphs using REBEL & Llama index** <br />
https://medium.com/@sauravjoshi23/building-knowledge-graphs-rebel-llamaindex-and-rebel-llamaindex-8769cf800115 <br />
**3)RAG based Knowledge graphs** <br />
https://medium.com/neo4j/enhancing-the-accuracy-of-rag-applications-with-knowledge-graphs-ad5e2ffab663  <br />
https://github.com/tomasonjo/blogs/blob/master/llm/enhancing_rag_with_graph.ipynb  <br />
https://github.com/Siddharthg97/Knowledge-graphs/blob/main/Theory.md  <br/>

**4)Prompt for Relationship extraction** <br />
https://github.com/sarthakrastogi/quality-prompts/blob/main/examples/few_shot_prompt_usage.ipynb <br />
https://api.python.langchain.com/en/latest/_modules/langchain_experimental/graph_transformers/llm.html <br />
https://simmering.dev/blog/structured_output/  <br />

**5) Langchain Chatmodels with systemprompttemplate,aipromptemplate,humanprompttemplate**
https://python.langchain.com/v0.1/docs/use_cases/extraction/
https://github.com/tomasonjo/blogs/blob/master/llm/enhancing_rag_with_graph.ipynb
https://www.youtube.com/watch?v=OagbDJvywJI
https://medium.com/@grisanti.isidoro/named-entity-recognition-with-llms-extract-conversation-metadata-94d5536178f2
llmner - chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://arxiv.org/pdf/2406.04528v1
https://api.python.langchain.com/en/latest/_modules/langchain_experimental/graph_transformers/llm.html

**6) Relationship extraction using trigger word and event argument extraction**

https://github.com/xinyadu/eeqa/tree/master?tab=readme-ov-file <br />
https://www.sciencedirect.com/science/article/pii/S266665102100005X <br />
https://www.mdpi.com/2076-3417/13/10/6308  <br />








