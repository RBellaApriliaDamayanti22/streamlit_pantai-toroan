import streamlit as st
import numpy as np
import pandas as pd
# import snscrape.modules.twitter as sntwitter
# import pandas as pd #pandas
# pd.options.mode.chained_assignment = None  # default='warn'
# import numpy as np #numpy
# import re #regex
# import string #string population
# import nltk
# from nltk.tokenize import word_tokenize #tokenize
# from nltk.corpus import stopwords #stopword
# from indoNLP.preprocessing import replace_slang #slank word
# from nltk.stem.porter import PorterStemmer #stemming

from PIL import Image

st.title("Aplikasi Informatika Pariwisata")

# inisialisasi data
tab1, tab2 = st.tabs(["Description data", "Processing"])

with tab1:
    st.subheader("Deskripsi")
    st.write(
        "Analisis Sentimen Terhadap Ulasan Komentar Pantai Toroan Kabupaten Sampang dengan Algoritma Naïve Bayes")
    img = Image.open("toroan.jpg")
    st.image(img)
    st.caption("""Air terjun menjadi salah satu pilihan destinasi untuk berwisata favorit selain pantai dan pegunungan. Pulau Madura yang dikenal sebagai pulau yang gersang sepertinya opini itu terbantahkan karena ternyata di Pulau Madura terdapat sebuah air terjun yang begitu indah. Air terjun Toroan Sampang adalah satu–satunya air terjun yang berada di Pulau Madura. Air terjun ini terletak di Kecamatan Ketapang, Kabupaten Sampang, Madura, Jawa Timur.""")

with tab2:
    st.subheader("Processing Data")
    comment = st.text_input('Write a comment')
    st.button("Check comment")
    st.write("Output")

    # Cache the dataframe so it's only loaded once
    @st.cache_data
    def load_data():
        return pd.DataFrame(
            {
                "first column": [],
                "second column": [],
            }
        )

    # Boolean to resize the dataframe, stored as a session state variable
    st.checkbox("Use container width", value=False, key="use_container_width")

    df = load_data()

    # Display the dataframe and allow the user to stretch the dataframe
    # across the full width of the container, based on the checkbox value
    st.dataframe(df, use_container_width=st.session_state.use_container_width)