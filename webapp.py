# import streamlit as st 
# import PyPDF2
# import pdfplumber
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# st.title("Candidate Selection Tool")

# st.subheader("NLP Based Resume Screening")

# st.caption("Aim of this project is to check whether a candidate is qualified for a role based his or her education, experience, and other information captured on their resume. In a nutshell, it's a form of pattern matching between a job's requirements and the qualifications of a candidate based on their resume.")

# uploadedJD = st.file_uploader("Upload Job Description", type="pdf")

# uploadedResume = st.file_uploader("Upload resume",type="pdf")

# click = st.button("Process")



# try:
#     global job_description
#     with pdfplumber.open(uploadedJD) as pdf:
#         pages = pdf.pages[0]
#         job_description = pages.extract_text()

# except:
#     st.write("")
    
    
# try:
#     global resume
#     with  pdfplumber.open(uploadedResume) as pdf:
#         pages = pdf.pages[0]
#         resume = pages.extract_text()
# except:
#     st.write("")
    
# #logic
# def getResult(JD_txt,resume_txt):
#     content = [JD_txt,resume_txt]

#     cv = CountVectorizer()

#     matrix = cv.fit_transform(content)

#     similarity_matrix =  cosine_similarity(matrix)

#     match = similarity_matrix[0][1] * 100

#     return match


# #button 

# if click:
#     match = getResult(job_description,resume)
#     match = round(match,2)
#     st.write("Match Percentage: ",match,"%")

# st.caption(" ~ made by villu")

import streamlit as st
import PyPDF2
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer  # Use TF-IDF for better keyword weighting
from sklearn.metrics.pairwise import cosine_similarity

st.title("Candidate Selection Tool")

st.subheader("NLP Based Resume Screening")

st.caption("This app checks whether a candidate is qualified for a role based on their resume and a job description, displaying both the match percentage and matched parameters.")

uploadedJD = st.file_uploader("Upload Job Description", type="pdf")
uploadedResume = st.file_uploader("Upload resume", type="pdf")

click = st.button("Process")

try:
    global job_description
    with pdfplumber.open(uploadedJD) as pdf:
        pages = pdf.pages[0]
        job_description = pages.extract_text()

except Exception as e:
    st.error("Error processing Job Description:", e)
    job_description = ""

try:
    global resume
    with pdfplumber.open(uploadedResume) as pdf:
        pages = pdf.pages[0]
        resume = pages.extract_text()

except Exception as e:
    st.error("Error processing Resume:", e)
    resume = ""

# Preprocess text (lowercase, remove stop words)
def preprocess_text(text):
    text = text.lower()
    stop_words = set(stopwords.words('english'))  # Use NLTK stopwords
    filtered_text = [word for word in text.split() if word not in stop_words]
    return " ".join(filtered_text)

# Tokenize text (split into words)
def tokenize_text(text):
    return text.split()

# Logic
def getResult(JD_txt, resume_txt):
    content = [JD_txt, resume_txt]

    # Use TF-IDF for better keyword weighting
    vectorizer = TfidfVectorizer(tokenizer=tokenize_text)
    matrix = vectorizer.fit_transform(content)

    similarity_matrix = cosine_similarity(matrix)
    match = similarity_matrix[0][1] * 100

    # Extract and display matched parameters
    matches = []
    JD_terms = vectorizer.get_feature_names_out()
    for i, score in enumerate(matrix[0]):
        if score > 0.5:  # Adjust threshold as needed
            matches.append(JD_terms[i])

    return match, matches

# Button
if click:
    if job_description and resume:
        job_description = preprocess_text(job_description)
        resume = preprocess_text(resume)

        match, matched_params = getResult(job_description, resume)
        match = round(match, 2)

        st.write("Match Percentage:", match, "%")

        if matched_params:
            st.subheader("Matched Parameters:")
            for param in matched_params:
                st.write(f"- {param}")
        else:
            st.info("No significant matches found.")
    else:
        st.error("Please upload both Job Description and Resume.")

st.caption("~ made by villu")

