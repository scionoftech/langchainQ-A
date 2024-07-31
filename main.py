import streamlit as st
import time
from prompt import get_results


def main():
    """
    Main function for running the Simple Q & A Application with OpenAI.
    """
    st.title("Simple Q & A Application with OpenAI")

    user_question = st.text_input("Type your question here:")

    if st.button("Submit"):
        with st.spinner("Requesting LLM for answer..."):
            # Call OpenAI API or perform any other logic here
            response, res_file, res_page = get_results(user_question)
            # Output box
            st.markdown("---")
            st.subheader("Output")

            # Output some results
            st.write("Results go here")
            st.write(response)
            st.write(f"Source Document: {res_file}")
            st.write(f"Page No: {res_page}")
            time.sleep(2)  # Simulating file processing time


if __name__ == "__main__":
    main()
