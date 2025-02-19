import streamlit as st
from pathlib import Path
import logging
from main import process_and_store_documents, query_documents

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_app():
    """Initialize the application and process documents if needed"""
    if 'initialized' not in st.session_state:
        process_and_store_documents()
        st.session_state.initialized = True

def main():
    st.set_page_config(
        page_title="MongoDB Document Search & QA",
        page_icon="🔍",
        layout="wide"
    )

    st.title("MongoDB Document Search & QA System")
    st.markdown("---")

    # Initialize app
    initialize_app()

    # Create sidebar for history
    with st.sidebar:
        st.header("Chat History")
        if 'history' not in st.session_state:
            st.session_state.history = []
        
        # Display chat history
        for i, (q, a) in enumerate(st.session_state.history):
            st.text_area(f"Q{i+1}", value=q, height=50, disabled=True)
            st.text_area(f"A{i+1}", value=a, height=100, disabled=True)
            st.markdown("---")

    # Main chat interface
    st.header("Ask a Question")
    
    # Query input
    query = st.text_area("Enter your question about MongoDB:", height=100)
    
    # Add a submit button
    if st.button("Submit Question"):
        if query:
            try:
                # Show spinner while processing
                with st.spinner("Searching documents and generating response..."):
                    # Create results directory if it doesn't exist
                    results_dir = Path("search_results")
                    results_dir.mkdir(exist_ok=True)
                    
                    # Generate results file name
                    results_file = results_dir / f"search_results_{query[:30].replace(' ', '_')}.txt"
                    
                    # Get response
                    response = query_documents(query, results_file)
                    
                    # Display response
                    st.markdown("### Answer:")
                    st.markdown(response["llm_response"])
                    
                    # Add to history
                    st.session_state.history.append((query, response["llm_response"]))
                    
                    # Show where results are saved
                    st.info(f"Detailed search results have been saved to: {results_file}")
                    
                    # Option to view raw context
                    with st.expander("View Retrieved Context"):
                        with open(results_file, 'r', encoding='utf-8') as f:
                            st.text(f.read())
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a question.")

    # Add some helpful information at the bottom
    st.markdown("---")
    st.markdown("""
    ### Tips:
    - Ask specific questions about MongoDB
    - Questions can be about architecture, performance, indexing, etc.
    - The system searches through PDF documents to find relevant information
    - Responses are generated based on the found context
    """)

if __name__ == "__main__":
    main() 