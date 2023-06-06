import tiktoken
import gradio as gr
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS
from langchain.llms import GPT4All, LlamaCpp
import os
import argparse



def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    print(encoding.encode(string))
    num_tokens = len(encoding.encode(string))
    return num_tokens

def summarize_pdf(pdf_file_path):
    loader = PyPDFLoader(pdf_file_path)
    docs = loader.load_and_split()
    chain = load_summarize_chain(llm=llm, chain_type="map_reduce")
    summary = chain.run(docs)   
    return summary


load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))
callbacks = [] # if args.mute_stream else [StreamingStdOutCallbackHandler()]

# Prepare the LLM 
#llm = OpenAI(temperature=0)
match model_type:
    case "LlamaCpp":
        llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, callbacks=callbacks, verbose=False)
    case "GPT4All":
        llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False)
    case _default:
        print(f"Model {model_type} not supported!")
        exit;

summarize = summarize_pdf('source_documents/2023_GPT4All_Technical_Report.pdf')
print(summarize)

faiss_index = FAISS.from_documents(pages, HuggingFaceEmbeddings())
docs = faiss_index.similarity_search("What makes a useful API?", k=2)
for doc in docs:
    print(str(doc.metadata["page"]) + ":", doc.page_content[:1000])
