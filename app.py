import os
import asyncio
import streamlit as st
from io import BytesIO
from docx import Document
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew
from crewai_tools import DirectorySearchTool

def save_pdf_file(uploaded_file, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    save_path = os.path.join(save_folder, uploaded_file.name)
    with open(save_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    return save_path

def tool(mod, api_key):
    if mod == 'Gemini':
        rag_tool = DirectorySearchTool(
            directory="Saved Files",
            config=dict(
                llm=dict(
                    provider="google",
                    config=dict(
                        model="gemini-1.5-flash",
                        temperature=0.6
                    ),
                ),
                embedder=dict(
                    provider="google",
                    config=dict(
                        model="models/embedding-001",
                        task_type="retrieval_document",
                        title="Embeddings"
                    ),
                ),
            )
        )
    else:

        rag_tool = DirectorySearchTool(
            directory="Saved Files",
            config=dict(
                llm=dict(
                    provider="openai",
                    config=dict(
                        model="gpt-3.5-turbo",
                        temperature=0.6
                    ),
                ),
                embedder=dict(
                    provider="openai",
                    config=dict(
                        model="text-embedding-ada-002",
                    ),
                ),
            )
        )
    return rag_tool

async def setup_openai(api_key):
    os.environ["OPENAI_API_KEY"] = api_key
    llm = ChatOpenAI(model='gpt-4-turbo', temperature=0.6, max_tokens=200)
    print("OpenAI Configured")
    return llm

async def setup_gemini(api_key):
    os.environ["GOOGLE_API_KEY"] = api_key
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        verbose=True,
        temperature=0.6,
        google_api_key=api_key
    )
    print("Gemini Configured")
    return llm

def generate_text(llm, question, rag_tool):
    inputs = {'question': question}

    writer_agent = Agent(
        role='Customer Service Specialist',
        goal='To accurately and efficiently answer customer questions',
        backstory='''
        As a seasoned Customer Service Specialist, this bot has honed its 
        skills in delivering prompt and precise solutions to customer queries.
        With a background in handling diverse customer needs across various industries,
        it ensures top-notch service with every interaction.
        ''',
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    task_writer = Task(
        description=(f'''Use the PDF RAG search tool to accurately and efficiently answer customer question. 
                     The customer question is : {question}
                     The task involves analyzing user queries and generating clear, concise, and accurate responses.'''),
        agent=writer_agent,
        expected_output="""
        - A detailed and well-sourced answer to customer's question.
        - Clear and concise synthesis of the retrieved information, formatted in a user-friendly manner.
        """,
        tools=[rag_tool]
    )

    crew = Crew(
        agents=[writer_agent],
        tasks=[task_writer],
        verbose=2,
        context={"Customer Question is ": question}
    )

    result = crew.kickoff(inputs=inputs)
    return result

st.header('RAG Content Generator')

with st.sidebar:
    with st.form('Gemini/OpenAI'):
        model = st.radio('Choose Your LLM', ['Gemini','Openai'])
        api_key = st.text_input(f'Enter your API key', type="password")
        submitted = st.form_submit_button("Submit")

mod = None

if api_key:
    if model == 'OpenAI':
        llm = asyncio.run(setup_openai(api_key))
        mod = 'OpenAI'
    elif model == 'Gemini':
        llm = asyncio.run(setup_gemini(api_key))
        mod = 'Gemini'

    rag_tool = tool(mod, api_key)
    question = st.text_input("Enter your question:")

    if st.button("Generate Answer"):
        with st.spinner("Generating Answer..."):
            generated_content = generate_text(llm, question, rag_tool)
            st.markdown(generated_content)

            # Uncomment the following lines if you want to provide a download button for the answer in a Word document.
            # doc = Document()
            # doc.add_heading(question, level=1)
            # doc.add_paragraph(generated_content)
            # buffer = BytesIO()
            # doc.save(buffer)
            # buffer.seek(0)
            # st.download_button(
            #     label="Download as Word Document",
            #     data=buffer,
            #     file_name=f"{question}.docx",
            #     mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            # )
