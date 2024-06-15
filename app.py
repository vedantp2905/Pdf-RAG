import os
import asyncio
import streamlit as st
from io import BytesIO
from docx import Document
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew
from crewai_tools import PDFSearchTool

def save_pdf_file(uploaded_file, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    save_path = os.path.join(save_folder, uploaded_file.name)
    with open(save_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    return save_path

def tool(path):
    rag_tool = PDFSearchTool(
        pdf=path, #path required.
        config=dict(
            
            llm=dict(
                provider="google",  # or google, openai, anthropic, llama2, ...
                config=dict(
                    model="gemini-1.5-flash",
                    temperature=0.6
                ),
            ),
            embedder=dict(
                provider="google",  # or openai, ollama, ...
                config=dict(
                    model="models/embedding-001",
                    task_type="retrieval_document",
                    title="Embeddings for PDF",
                    
                ),
            ),
        )
    )
    return rag_tool


def generate_text(llm, question, rag_tool):
    inputs = {'question': question}

    
    
    writer_agent = Agent(
        role='Research Specialist',
        goal='To accurately and efficiently retrieve and synthesize information from PDFs to answer user questions.',
        backstory='''
        Alex has always been passionate about research and information synthesis. With a background in library science 
        and information technology, Alex developed a keen eye for finding precise information in vast databases and documents.
        Joining the AI research team, Alex now uses advanced PDF retrieval tools to help users find accurate answers quickly.
        Alex's meticulous nature and dedication to accuracy make them the perfect fit for the role of Research Specialist, ensuring
        that every question is answered with reliable and well-sourced information.
        ''',
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    task_writer = Task(
        description=('''Use the PDF RAG search tool to accurately and efficiently retrieve and synthesize information from provided PDFs to answer user questions. 
                     The task involves analyzing user queries, retrieving relevant information from the PDFs, and generating clear, concise, and accurate responses.'''),
        agent=writer_agent,
        expected_output="""
        - A detailed and well-sourced answer to the user's question.
        - References to specific sections or pages in the PDFs where the information was found.
        - Clear and concise synthesis of the retrieved information, formatted in a user-friendly manner.
        """,
        tools=[rag_tool]
    )

    crew = Crew(
        agents=[writer_agent],
        tasks=[task_writer],
        verbose=2
    )

    result = crew.kickoff(inputs=inputs)
    return result

def main():
    st.header('RAG Content Generator')
    mod = None

    with st.sidebar:
        with st.form('Gemini/OpenAI'):
            model = st.radio('Choose Your LLM', ('Gemini', 'OpenAI'))
            api_key = st.text_input(f'Enter your API key', type="password")
            submitted = st.form_submit_button("Submit")

    if api_key and submitted:
        if model == 'OpenAI':
            async def setup_OpenAI():
                os.environ["OPENAI_API_KEY"] = api_key
                llm = ChatOpenAI(temperature=0.6, max_tokens=2000)
                print("OpenAI Configured")
                return llm

            llm = asyncio.run(setup_OpenAI())
            mod = 'OpenAI'

        elif model == 'Gemini':
            async def setup_gemini():
                
                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    verbose=True,
                    temperature=0.6,
                    google_api_key=api_key
                )
                print("Gemini Configured")
                return llm

            llm = asyncio.run(setup_gemini())
            mod = 'Gemini'

    uploaded_file = st.file_uploader("Choose a PDF file", type='pdf')

    if uploaded_file is not None:
        
        save_folder = 'Saved Files'
        saved_path = save_pdf_file(uploaded_file, save_folder)
        
        rag_tool = tool(saved_path)
        
        question = st.text_input("Enter your question:")
        

        if st.button("Generate Answer"):
            with st.spinner("Generating Answer..."):
                
                generated_content = generate_text(llm, question,rag_tool)

                st.markdown(generated_content)

                doc = Document()
                doc.add_heading(question, level=1)  # Specify the heading level
                doc.add_paragraph(generated_content)

                buffer = BytesIO()
                doc.save(buffer)
                buffer.seek(0)
                
                st.download_button(
                        label="Download as Word Document",
                        data=buffer,
                        file_name=f"{question}.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )

if __name__ == "__main__":
    main()
