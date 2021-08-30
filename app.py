import faiss
import pickle
import pandas as pd
import streamlit as st
import time
import numpy as np
from sentence_transformers import SentenceTransformer


@st.cache
def read_data(data="data/Data.csv"):
    """Read the data from local."""
    return pd.read_csv(data)


@st.cache(allow_output_mutation=True)
def load_bert_model(name="msmarco-distilbert-base-dot-prod-v3"):
    """Instantiate a sentence-level DistilBERT model."""
    return SentenceTransformer(name)


@st.cache(allow_output_mutation=True)
def load_faiss_index(path_to_faiss="model/faiss_index.pickle"):
    """Load and deserialize the Faiss index."""
    with open(path_to_faiss, "rb") as h:
        data = pickle.load(h)
    return faiss.deserialize_index(data)

def fetch_report_info(dataframe_idx, pak_df):
    info = pak_df.iloc[dataframe_idx]
    meta_dict = {}
    meta_dict['text'] = info['Text']
    return meta_dict

def search(query, top_k, index_pak, model):
    t=time.time()
    query_vector = model.encode([query])
    top_k = index_pak.search(query_vector, top_k)
    print('>>>> Results in Total Time: {}'.format(time.time()-t))
    # top_k_ids = top_k[1].tolist()[0]
    # top_k_ids = list(np.unique(top_k_ids))
    # results =  [fetch_report_info(idx,) for idx in top_k_ids]
    return top_k



def main():
    # Data and model
    data = read_data()
    model = load_bert_model()
    index_pak = load_faiss_index()

    st.title("Semantic Search Engine with SentenceTransformer and FAISS")

    # User search
    user_input = st.text_area("Search box", "Muder")
    
    # Filter
    num_results = st.sidebar.slider("Number of search results", 10, 50, 5)

    # Results
    top_k = search(user_input, top_k=num_results, index_pak=index_pak, model=model)
    top_k_ids = top_k[1].tolist()[0]
    top_k_ids = list(np.unique(top_k_ids))
    results =  [fetch_report_info(idx,data) for idx in top_k_ids]

    for i in range(len(results)):
        st.write(f"{results[i].get('text')}")
        st.markdown("***")

if __name__ == "__main__":
    main()