import requests
import streamlit as st

API_URL = "http://api:8000/ask"

st.set_page_config(
    page_title="Persian Q-A RAG", 
    layout="wide")

st.title("Persian Q-A RAG Asistant")
st.write("Ask a question based on dataset.")

query = st.text_input("Ask a question:")

k = st.slider(
    "Number of retrieved documents",
    min_value = 1,
    max_value = 5,
    value =3
)

if st.button("Ask") and query:
    with st.spinner("sending your request to API..."):
        response = requests.post(
            API_URL,
            json={'question': query, "k": k},
            timeout = 180
        )
    
    if response.status_code == 200:
        result = response.json()
        
        st.subheader("Answer:")
        st.write(result['answer'])
        
        st.subheader("Retrieved Documents:")
        for source in result['sources']:
            st.json(source)

    else:
        st.error(f"API Error: {response.status_code}")
        st.write(response.text)