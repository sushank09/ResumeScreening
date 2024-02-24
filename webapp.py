import streamlit as st
import PyPDF2
import pdfplumber
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

st.title("Candidate Selection Tool")

st.subheader("NLP Based Resume Screening")

st.caption("Aim of this project is to check whether a candidate is qualified for a role based his or her education, experience, and other information captured on their resume. In a nutshell, it's a form of pattern matching between a job's requirements and the qualifications of a candidate based on their resume.")

uploadedJD = st.file_uploader("Upload Job Description", type="pdf")

uploadedResume = st.file_uploader("Upload resume",type="pdf")

click = st.button("Process")

try:
    global job_description
    with pdfplumber.open(uploadedJD) as pdf:
        pages = pdf.pages[0]
        job_description = pages.extract_text()

except:
    st.write("")
    
    
try:
    global resume
    with  pdfplumber.open(uploadedResume) as pdf:
        pages = pdf.pages[0]
        resume = pages.extract_text()
except:
    st.write("")
    
#logic
def getResult(JD_txt,resume_txt):
    content = [JD_txt,resume_txt]

    cv = CountVectorizer()

    matrix = cv.fit_transform(content)

    similarity_matrix =  cosine_similarity(matrix)

    match = similarity_matrix[0][1] * 100

    return match, similarity_matrix, cv


#button 

if click:
    match, similarity_matrix, cv = getResult(job_description,resume)
    match = round(match,2)
    st.write("Match Percentage: ", match, "%")
    st.write("Details of matching data:")
    st.write("Cosine Similarity Matrix:")
    st.write(similarity_matrix)
    st.write("Matching Keywords:")
    job_keywords = cv.get_feature_names_out()
    matching_keywords = []
    for idx, val in enumerate(similarity_matrix[0]):
        if val > 0:
            matching_keywords.append(job_keywords[idx])
    highlighted_resume = resume
    for keyword in matching_keywords:
        highlighted_resume = highlighted_resume.replace(keyword, f"<mark>{keyword}</mark>")
    st.write("Highlighted Resume:")
    st.markdown(highlighted_resume, unsafe_allow_html=True)

    # Adding a bar chart for visualization
    fig, ax = plt.subplots()
    ax.bar(["Match Percentage", "Difference"], [match, 100-match], color=['blue', 'red'])
    ax.set_ylabel('Percentage')
    ax.set_title('Similarity Comparison')
    st.pyplot(fig)

st.caption(" ~ made by team")

