import streamlit as st
import pandas as pd
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
#from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
from langchain_community.llms import AlephAlpha
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.embeddings.aleph_alpha import AlephAlphaAsymmetricSemanticEmbedding, AlephAlphaSymmetricSemanticEmbedding

load_dotenv()

def load_process_data():
    # Load data from different sources
    df_csv = pd.read_csv('data/old_tickets/ticket_dump_1.csv')  # CSV file
    df_excel = pd.read_excel('data/old_tickets/ticket_dump_2.xlsx')  # Excel file
    df_json = pd.read_json('data/old_tickets/ticket_dump_3.json')  # JSON file

    # Combine the DataFrames
    df_combined = pd.concat([df_csv, df_excel, df_json], ignore_index=True)

    # Filter for resolved tickets only
    df_combined = df_combined[df_combined['Resolved'] == True]


    df_combined['combined_text'] = df_combined.apply(lambda row: f"Issue: {row['Issue']}\nDescription: {row['Description']}", axis=1)

    #print("Document Shape: ",df_combined.shape)

    # Prepare documents with metadata including Category
    documents = [
        Document(
            page_content=text, 
            metadata={
                "Ticket ID": row['Ticket ID'], 
                "Resolution": row['Resolution'], 
                "Category": row['Category']  # Include Category in metadata
            }
        ) 
        for text, row in zip(df_combined['combined_text'], df_combined.to_dict('records'))
    ]
    return documents

# Define the retrieval and generation pipeline with hybrid search
def get_suggestions(chain, retriever, issue, description, category):
    # Combine issue and description to match the vectorized documents
    query = f"Issue: {issue}\nDescription: {description}"
    
    # Retrieve similar tickets within the same category
    #similar_docs = retriever.get_relevant_documents(query, metadata_filters={"Category": category})
    similar_docs = retriever.invoke(query, metadata_filters={"Category": category})
    # Extract resolutions from the retrieved documents
    resolutions = [doc.metadata['Resolution'] for doc in similar_docs]
    
    # Combine resolutions into a single context for the LLM
    combined_resolutions = "\n".join(resolutions)
    
    # Generate the suggestion with combined_resolutions
    suggestion = chain.invoke({"issue": issue, "description": description, "combined_resolutions": combined_resolutions})
    
    return suggestion


# Example Usage
#new_ticket_issue = "VPN not connecting"
#new_ticket_description = "VPN fails to connect to the company's network, showing error messages."
#new_ticket_category = "Network"

#suggestion = get_suggestions(chain, retriever, new_ticket_issue, new_ticket_description, new_ticket_category)
#print(suggestion)

if __name__=="__main__":

    embeddings = AlephAlphaAsymmetricSemanticEmbedding(normalize=True,compress_to_size=128, aleph_alpha_api_key=os.getenv("AA_TOKEN"))
    #documents = load_process_data()
    #chroma_db = Chroma.from_documents(documents=documents, embedding=embeddings,persist_directory="./chroma_db_aleph")
    #print(len(chroma_db.get()["ids"]))

    chroma_db = Chroma(persist_directory="./chroma_db_aleph", embedding_function=embeddings)

    #print("Chroma_DB_Documents: ",len(chroma_db.get()["ids"]))

    retriever = chroma_db.as_retriever(search_kwargs={'k': 1})

    prompt_template = """
    You are an IT helpdesk assistant. A new ticket has been raised with the following issue and description:

    Issue: {issue}
    Description: {description}

    Here are some previous resolutions that might be relevant to this issue:
    {combined_resolutions}

    Based on the above information, provide a direction or suggestion that might help in resolving this issue. The goal is not to provide an exact solution but to give a hint or direction that could be helpful.
    """

    prompt = PromptTemplate(input_variables=["issue", "description", "combined_resolutions"], template=prompt_template)

    llm = AlephAlpha(model="luminous-extended-control", maximum_tokens=256, aleph_alpha_api_key=os.getenv("AA_TOKEN"))

    chain = prompt | llm

    # Streamlit app interface
    st.title('IT Helpdesk Assistant')

    # Input fields for new ticket
    issue = st.text_input("Issue")
    description = st.text_area("Description")
    category_list = ['Software', 'Network', 'Account Management', 'Hardware']
    category = st.selectbox("Category", category_list)

    # Button to generate suggestions
    if st.button("Get Suggestions"):
        if issue and description and category:
            suggestion = get_suggestions(chain, retriever, issue, description, category)
            st.subheader("Suggested Resolution Direction:")
            st.write(suggestion)
        else:
            st.error("Please provide an issue, description, and select a category.")