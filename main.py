import streamlit as st

st.set_page_config(
    page_title="L'Oréal Datathon",
    layout="wide",
    menu_items={
        'About': """
        This is the website for L'Oréal Datathon.
        """
    }
)
st.logo("https://www.loreal.com/-/media/project/loreal/brand-sites/corp/master/lcorp/7-local-country-folder/malaysia/news/group/scam-whatsapp-message-on-loreal/loreal-malaysia-logo01.jpg?rev=deb16ca878f4468e81d46554f7334b34&cx=0.49&cy=0.5&cw=1920&ch=800&blr=False&hash=9435E8E651AD6717C4A582E6D653E5B1", link="https://www.loreal.com/en/malaysia/", size="large")

pages = [
 st.Page("pages/dashboard.py", title="Dashboard", icon=":material/analytics:", default=True),
 st.Page("pages/analyzeComment.py", title="Comment Senser", icon=":material/video_chat:"),
 st.Page("pages/chatbot.py", title="Loreal Chat", icon=":material/robot_2:"),
]

# --- NAVIGATION SETUP ---
main = st.navigation(pages, position="sidebar")
main.run()
