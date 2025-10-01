import streamlit as st
import pandas as pd
from src.youtube_scrapper import scrape_video
from src.report_generator import generate_pdf_report  # adjust import if needed

st.title("YouTube Video & Comment Scraper")

url_input = st.text_input("Enter YouTube Video URL:")

if url_input:
    if st.button("Scrape & Generate Report"):
        with st.spinner("Scraping video and comments..."):
            df = scrape_video(url_input)

            if df.empty:
                st.error("No data found. Check the URL or API key.")
            else:
                st.dataframe(df)

                with st.spinner("Generating PDF report..."):
                    pdf_report = generate_pdf_report(df)
                    video_title = df['title'].iloc[0].replace(' ', '_')
                    st.download_button(
                        label="Download Report",
                        data=pdf_report,
                        file_name=f"report_{video_title}.pdf",
                        mime="application/pdf"
                    )
