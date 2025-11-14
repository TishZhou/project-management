import os
import openai
# from os import environ
import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter  # <-- FIX 1: Corrected spelling
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough  # <-- FIX 2: Corrected 'runnables'
from PyPDF2 import PdfReader
from langchain_experimental.text_splitter import SemanticChunker
from langchain.vectorstores import FAISS
from jobspy import scrape_jobs
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
# Streamlit setup
st.title("AI Job Assistant")

# Set up OpenAI proxy client
client = openai.OpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
    base_url=os.getenv('OPENAI_BASE_URL', 'https://api.ai.it.cornell.edu/')
)

# initialize session state for job search results and selected job
if "jobs_df" not in st.session_state:
    st.session_state["jobs_df"] = None
if "selected_job" not in st.session_state:
    st.session_state["selected_job"] = None
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! Upload your resume to see how you match with this job."}]

# function to search jobs
def search_jobs(search_term, location, job_type, days_old):
    try:
        jobs = scrape_jobs(
            site_name=["indeed", "linkedin", "glassdoor", "google"],
            search_term=search_term,
            location=location,
            job_type=job_type,
            hours_old=days_old * 24  # convert days to hours
        )
        return jobs
    except Exception as e:
        st.error(f"Error searching jobs: {str(e)}")
        return pd.DataFrame()

# show job search interface if no job is selected
if st.session_state["selected_job"] is None:
    st.header("Job Search")
    
    # job search form
    with st.form("job_search_form"):
        search_term = st.text_input("Job Title", "Software Engineer")
        location = st.text_input("Location", "New York, NY")
        job_type = st.selectbox(
            "Job Type",
            [None, "Full-time", "Part-time", "Internship", "Contract"]
        )
        days_old = st.slider("Posted Within (days)", 1, 30, 7)
        
        # --- MODIFICATION 1: ADDED CHECKBOX ---
        sponsor_filter = st.checkbox("Only show jobs mentioning H1B/Sponsorship")
        # ----------------------------------------
        
        search_submitted = st.form_submit_button("Search Jobs")
        
        if search_submitted:
            with st.spinner("Searching jobs across LinkedIn, Indeed, Glassdoor, and Google..."):
                jobs_df = search_jobs(
                    search_term=search_term,
                    location=location,
                    job_type=job_type,
                    days_old=days_old
                )
                
                # --- MODIFICATION 2: ADDED FILTER LOGIC ---
                if sponsor_filter and not jobs_df.empty:
                    with st.spinner("Filtering for H1B/Sponsorship..."):
                        # Define keywords
                        sponsor_keywords = ['h1b', 'h-1b', 'visa', 'sponsorship', 'sponsor']
                        negation_keywords = [
                            'no h1b', 'not sponsor', 'will not sponsor', 'does not sponsor',
                            'unable to sponsor', 'no sponsorship', 'sponsorship is not available'
                        ]

                        # Prepare description column for search
                        desc_lower = jobs_df['description'].str.lower().fillna('')
                        
                        # Create "Positive" Mask (must contain a sponsor keyword)
                        positive_mask = desc_lower.str.contains('|'.join(sponsor_keywords), na=False)
                        
                        # Create "Negative" Mask (must NOT contain a negation keyword)
                        negative_mask = desc_lower.str.contains('|'.join(negation_keywords), na=False)
                        
                        # Apply the final mask: Must be Positive AND NOT Negative
                        final_mask = positive_mask & ~negative_mask
                        jobs_df = jobs_df[final_mask]
                # --- END OF FILTER LOGIC ---

                if not jobs_df.empty:
                    st.session_state["jobs_df"] = jobs_df
                    st.success(f"Found {len(jobs_df)} jobs!")
                else:
                    st.error("No jobs found or error occurred during search.")

    # display job search results if available
    if st.session_state["jobs_df"] is not None:
        st.subheader("Search Results")
        
        # create an expandable section for each job
        for idx, job in st.session_state["jobs_df"].iterrows():
            # handle job details - not all jobs have details
            with st.expander(f"{job['title']} at {job['company']}"):
                # handle location display
                location_str = job.get('location', 'Location not specified')
                st.write(f"**Location:** {location_str}")
                
                # handle job type display
                raw_job_type = job.get('job_type', None)
                if pd.isna(raw_job_type) or str(raw_job_type).lower() == 'nan':
                    job_type_display = 'Not provided'
                else:
                    # re-map job types
                    job_type_mappings = {
                        'fulltime': 'Full-time',
                        'parttime': 'Part-time',
                        'internship': 'Internship',
                        'contract': 'Contract'
                    }
                    job_type_display = job_type_mappings.get(str(raw_job_type).lower(), str(raw_job_type))
                st.write(f"**Job Type:** {job_type_display}")

                # handle salary display
                min_amount = job.get('min_amount')
                max_amount = job.get('max_amount')

                # convert 'nan' strings or numpy NaN to None
                if pd.isna(min_amount) or str(min_amount).lower() == 'nan':
                    min_amount = None
                if pd.isna(max_amount) or str(max_amount).lower() == 'nan':
                    max_amount = None

                # display salary
                if min_amount is not None and max_amount is not None:
                    formatted_min = "{:,.0f}".format(float(min_amount))
                    formatted_max = "{:,.0f}".format(float(max_amount))
                    st.write(f"**Salary Range:** {formatted_min} - {formatted_max}")
                elif min_amount is not None:
                    formatted_min = "{:,.0f}".format(float(min_amount))
                    st.write(f"**Salary:** {formatted_min}+")
                elif max_amount is not None:
                    formatted_max = "{:,.0f}".format(float(max_amount))
                    st.write(f"**Salary:** Up to {formatted_max}")
                else:
                    st.write("**Salary:** Not provided")
                                
                # display job description
                description = job.get('description', 'No description available')
                st.write("**Description:**")
                st.write(description[:500] + "..." if len(description) > 500 else description)
                
                if st.button("Select This Job", key=f"select_job_{idx}"):
                    st.session_state["selected_job"] = job
                    st.rerun()
                    
# show chat interface if a job is selected
else:
    # back button to return to job search
    if st.button("← Back to Job Search"):
        st.session_state["selected_job"] = None
        st.rerun()
        
    # display selected job details
    st.header(f"Selected Job: {st.session_state['selected_job']['title']}")
    st.subheader(f"Company: {st.session_state['selected_job']['company']}")
    
    # file uploader for resume
    uploaded_files = st.file_uploader("Upload your resume", type=("txt", "pdf"), accept_multiple_files=True)

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    question = st.chat_input("Ask a question about this job")

    # Clear system cache if needed
    # chromadb.api.client.SharedSystemClient.clear_system_cache()

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        st.chat_message("user").write(question)

        if uploaded_files:
            combined_text = ""
            for uploaded_file in uploaded_files:
                if uploaded_file.name.endswith(".pdf"):
                    pdf_reader = PdfReader(uploaded_file)
                    for page in pdf_reader.pages:
                        combined_text += page.extract_text()
                elif uploaded_file.name.endswith(".txt"):
                    combined_text += uploaded_file.read().decode("utf-8")

            # Vectorstore and context setup
            documents = [Document(page_content=combined_text)]
            embeddings = OpenAIEmbeddings(
                model="openai.text-embedding-3-large",
                openai_api_key=os.getenv('OPENAI_API_KEY'),
                openai_api_base=os.getenv('OPENAI_BASE_URL', 'https://api.ai.it.cornell.edu/')
            )
            vectorstore = FAISS.from_documents(documents, embeddings)
            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

            # Prompt setup
            template = """
            You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
            If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

            Question: {question} 

            Context: {context} 

            Answer:
            """
            prompt = PromptTemplate.from_template(template)

            # Build the RAG chain
            rag_chain = (
                {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                | prompt
                | client.chat.completions.create  # Use chat completion API
                | StrOutputParser()
            )

            # Retrieve and process documents
            relevant_docs = retriever.get_relevant_documents(question)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])

            # Generate response
            messages = {
                "model": "openai.gpt-4o",  # Specify the model
                "messages": [{"role": "user", "content": f"Question: {question}\n\nContext: {context}"}],
            }
            response = client.chat.completions.create(**messages)

            st.chat_message("assistant").write(response.choices[0].message.content)
            st.session_state.messages.append({"role": "assistant", "content": response.choices[0].message.content})
            
        else:
            # Direct response without context
            messages = {
                "model": "openai.gpt-4o",
                "messages": [{"role": "user", "content": question}],
            }
            response = client.chat.completions.create(**messages)
            st.chat_message("assistant").write(response.choices[0].message.content)
            st.session_state.messages.append({"role": "assistant", "content": response.choices[0].message.content})