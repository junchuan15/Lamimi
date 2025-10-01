import streamlit as st
import pandas as pd
import altair as alt
import os
from src.youtube_scrapper import scrape_video
from src.report_generator import generate_pdf_report, generate_video_summary

def comment_senser_page():
    # --- Load CSS ---
    css_path = os.path.join(os.path.dirname(__file__), "../styles/analyze_comment_page.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # --- Header ---
    st.markdown(
        """
        <div class="analyze-header">
            <h1>Youtube Video & Comment Scrapper</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- Initialize session state ---
    if "video_url" not in st.session_state:
        st.session_state.video_url = ""
    if "video_data" not in st.session_state:
        st.session_state.video_data = None
    if "video_summary" not in st.session_state:
        st.session_state.video_summary = None
    if "pdf_report" not in st.session_state:
        st.session_state.pdf_report = None
        
    st.markdown('<div class="form-label"><h5>Insert Video/Shorts Link For Analysis</h5></div>', unsafe_allow_html=True)
    url_input = st.text_input(
        "",
        label_visibility="collapsed",
        value=st.session_state["video_url"],
        placeholder="Enter YouTube Video/Shorts URL"
    )

    def process_video(url):
        """Scrape video, generate summary, and PDF report."""
        df = scrape_video(url)
        if df.empty:
            st.error("No data found. Check the URL or API key.")
            return None, None, None

        # Video summary
        video_title = df['title'].iloc[0]
        video_description = df['description'].iloc[0]
        summary = generate_video_summary(video_title, video_description)

        # PDF report
        pdf_report = generate_pdf_report(df)
        return df, summary, pdf_report

    if url_input:
        if st.button("Scrape & Generate Report"):
            df, summary, pdf_report = process_video(url_input)
            st.session_state.video_data = df
            st.session_state.video_summary = summary
            st.session_state.pdf_report = pdf_report

    # --- Display if data exists ---
    if st.session_state.video_data is not None:
        df = st.session_state.video_data
        summary = st.session_state.video_summary
        pdf_report = st.session_state.pdf_report

        # Video Details
        video_title = df['title'].iloc[0]
        video_description = df['description'].iloc[0]
        video_channel_id = df['channelId'].iloc[0]
        video_published_date = df['publishedAt'].iloc[0]
        video_likes = df['likeCount'].iloc[0]
        video_engagement_rate = df['engagement_rate'].iloc[0]

        st.markdown("<div class='video-details'>", unsafe_allow_html=True)
        st.markdown(f"<p class='video-title'>{video_title}</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='video-description'>{video_description}</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='video-summary'><strong>Summary:</strong> {summary}</p>", unsafe_allow_html=True)
        st.markdown(f"""
            <p><strong>Channel ID:</strong> {video_channel_id}</p>
            <p><strong>Published Date:</strong> {video_published_date}</p>
            <p><strong>Likes:</strong> {video_likes:,}</p>
            <p><strong>Engagement Rate:</strong> {video_engagement_rate:.2%}</p>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # --- Comment Analysis Overview ---
        st.subheader("Comment Analysis Overview")

        total_comments = len(df)
        spam_count = df['is_spam'].sum()
        spam_ratio = spam_count / total_comments if total_comments > 0 else 0
        quality_comments = len(df[df['sentiment_score'] >= 0.7])

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Comments", f"{total_comments:,}")
        col2.metric("Spam Ratio", f"{spam_ratio:.2%}")
        col3.metric("Quality Comments", f"{quality_comments:,}")

        # --- Charts ---
        col1, col2 = st.columns(2)
        with col1:
            sentiment_counts = df['sentiment_label'].value_counts().reset_index()
            sentiment_counts.columns = ['sentiment_label', 'count']
            sentiment_chart = alt.Chart(sentiment_counts).mark_arc(innerRadius=50).encode(
                theta=alt.Theta(field="count", type="quantitative"),
                color=alt.Color(field="sentiment_label", type="nominal", title="Sentiment"),
                tooltip=['sentiment_label', 'count']
            ).properties(title="Sentiment Distribution")
            st.altair_chart(sentiment_chart, use_container_width=True)

        with col2:
            category_counts = df['cluster_label'].value_counts().nlargest(5).reset_index()
            category_counts.columns = ['cluster_label', 'count']
            category_chart = alt.Chart(category_counts).mark_bar().encode(
                x=alt.X('cluster_label', sort='-y', title="Product Category"),
                y=alt.Y('count', title="Number of Comments"),
                tooltip=['cluster_label', 'count']
            ).properties(title="Top 5 Product Categories by Comment Volume")
            st.altair_chart(category_chart, use_container_width=True)

        sentiment_by_category = df.groupby(['cluster_label', 'sentiment_label']).size().reset_index(name='count')
        sentiment_category_chart = alt.Chart(sentiment_by_category).mark_bar().encode(
            x=alt.X('cluster_label', title="Product Category"),
            y=alt.Y('count', title="Number of Comments"),
            color=alt.Color('sentiment_label', title="Sentiment"),
            tooltip=['cluster_label', 'sentiment_label', 'count']
        ).properties(title="Sentiment Breakdown by Product Category")
        st.altair_chart(sentiment_category_chart, use_container_width=True)

        # --- PDF download ---
        video_title_safe = "".join([c if c.isalnum() else "_" for c in video_title])
        st.download_button(
            label="Download Report",
            data=pdf_report,
            file_name=f"report_{video_title_safe}.pdf",
            mime="application/pdf"
        )
